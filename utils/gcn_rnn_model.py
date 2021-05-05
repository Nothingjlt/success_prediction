from networkx.algorithms.assortativity import correlation
import torch
from torch_geometric.data import Data
from torch import nn, optim
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import nni
from utils.gcn_rnn_net import GCNRNNNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, nni, parameters):
        self._params = parameters
        self._device = DEVICE
        self._gcn_data_list = None
        # self._criterion = nn.L1Loss()
        self._criterion = nn.MSELoss()
        # self._accuracy_metric = nn.L1Loss()
        self.train_idx = None
        self.test_idx = None
        self.validation_idx = None
        self._gcn_rnn_net = None
        self._optimizer = None
        self._print_every_num_of_opochs = parameters['print_every_num_of_epochs']
        self._nni = nni

    def load_data(
        self,
        gnxs: list,
        feature_matrix,
        current_timestamp_labels=None,
        next_timestamp_labels=None,
        learned_label: str = "",
        train=None,
        test=None,
        validation=None,
        accuracy_metric=r2_score,
    ):
        self._gcn_data_list = []

        self._learned_label = learned_label

        self._accruacy_metric = accuracy_metric

        all_nodes_set = set()
        for gnx in gnxs:
            all_nodes_set.update(gnx.nodes())
        all_nodes_list = sorted(all_nodes_set)

        self._num_of_nodes = len(all_nodes_list)

        self._node_id_to_idx = {x: i for i, x in enumerate(all_nodes_list)}
        self._idx_to_node_id = {i: x for i, x in enumerate(all_nodes_list)}

        for gnx in gnxs:
            gnx.add_nodes_from(all_nodes_set)
            nodes = gnx.nodes()
            x = torch.tensor(
                np.vstack([feature_matrix[self._node_id_to_idx[node]]
                           for node in nodes]),
                device=self._device,
            )

            edges = torch.tensor(
                np.vstack(
                    [
                        [self._node_id_to_idx[x[0]] for x in gnx.edges],
                        [self._node_id_to_idx[x[1]] for x in gnx.edges],
                    ]
                ),
                dtype=torch.long,  # Required by torch_geometric.data.Data
                device=self._device,
            )

            d = Data(x=x, edge_index=edges)

            self._gcn_data_list.append(d)

        self.train_idx = [self._node_id_to_idx[node] for node in train]
        self.test_idx = [self._node_id_to_idx[node] for node in test]
        self.validation_idx = [self._node_id_to_idx[node]
                               for node in validation]

        self.train_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self.train_idx)
        self.train_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self.train_idx)

        self.test_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self.test_idx)
        self.test_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self.test_idx)

        self.validation_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self.validation_idx)
        self.validation_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self.validation_idx)

        # Special case where both in_deg and out_deg are learned, reduce to deg.
        if self._learned_label == "general":

            # Trials with Yoram
            self.train_labels_to_learn = self.train_next_time_labels - self.train_current_labels
            self.test_labels_to_learn = self.test_next_time_labels - self.test_current_labels
            self.validation_labels_to_learn = self.validation_next_time_labels - \
                self.validation_current_labels

            # End trials with Yoram
            # self.train_next_time_labels = self.train_next_time_labels.sum(
            #     dim=1).log()
            # self.test_next_time_labels = self.test_next_time_labels.sum(
            #     dim=1).log()
            # self.validation_next_time_labels = self.validation_next_time_labels.sum(
            #     dim=1).log()

        self.train_idx = [self._node_id_to_idx[node] for node in train]
        self.test_idx = [self._node_id_to_idx[node] for node in test]
        self.validation_idx = [self._node_id_to_idx[node]
                               for node in validation]

        self._gcn_rnn_net = GCNRNNNet(
            feature_matrix.shape[1],
            self._params["gcn_latent_dim"],
            self._params["gcn_hidden_sizes"],
            self._params["gcn_dropout_rate"],
            self._params["lstm_hidden_size"],
            self._params["lstm_num_layers"],
            self._params["lstm_dropout_rate"],
        )

        self._optimizer = optim.Adam(
            self._gcn_rnn_net.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=self._params["weight_decay"],
        )

        self._l1_lambda = self._params["l1_lambda"]

        print(
            f"number of params in model: {sum(p.numel() for p in self._gcn_rnn_net.parameters())}")

        return

    def _get_labels_by_indices(self, labels, indices):
        ret_labels = torch.tensor(
            [labels[self._learned_label].features[self._idx_to_node_id[i]]
                for i in indices],
            dtype=torch.float,
            device=self._device,
        )
        # Special case where rank is required
        if self._learned_label == "general":
            if ret_labels.dim() == 2:
                ret_labels = ret_labels.sum(dim=1)
            ret_labels = torch.log(ret_labels)
        return ret_labels

    @property
    def data(self):
        return self._gcn_data_list.clone()

    def _ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def _l1_norm(self):
        l1_norm = 0
        for p in self._gcn_rnn_net.parameters():
            l1_norm += p.abs().sum()
        return l1_norm

    def evaluate(self, evaluation_set: str = "test", evaluate_accuracy: bool = False, evaluate_correlation: bool = False):
        # self._gcn_rnn_net.eval()

        if evaluation_set == "test":
            self._gcn_rnn_net.eval()
            evaluation_set_idx = self.test_idx
            # Trials with Yoram
            # evaluation_set_labels = self.test_next_time_labels
            evaluation_set_learned_labels = self.test_labels_to_learn
            evaluation_set_current_labels = self.test_current_labels
            evaluation_set_next_labels = self.test_next_time_labels
            # End trial with Yoram
        elif evaluation_set == "validation":
            self._gcn_rnn_net.eval()
            evaluation_set_idx = self.validation_idx
            # Trials with Yoram
            # evaluation_set_labels = self.validation_next_time_labels
            evaluation_set_learned_labels = self.validation_labels_to_learn
            evaluation_set_current_labels = self.validation_current_labels
            evaluation_set_next_labels = self.validation_next_time_labels
            # End trial with Yoram
        elif evaluation_set == "train":
            evaluation_set_idx = self.train_idx
            # Trials with Yoram
            # evaluation_set_labels = self.train_next_time_labels
            evaluation_set_learned_labels = self.train_labels_to_learn
            evaluation_set_current_labels = self.train_current_labels
            evaluation_set_next_labels = self.train_next_time_labels
            # End trial with Yoram
        # Default to test+validation
        elif evaluation_set == "testval":
            self._gcn_rnn_net.eval()
            evaluation_set_idx = self.validation_idx + self.test_idx
            evaluation_set_learned_labels = torch.cat(
                (self.validation_labels_to_learn, self.test_labels_to_learn))
            evaluation_set_current_labels = torch.cat(
                (self.validation_current_labels, self.test_current_labels))
            evaluation_set_next_labels = torch.cat(
                (self.validation_next_time_labels, self.test_next_time_labels))
        else:
            print(f"Invalid evaluation set: {evaluation_set}")
            return

        val_output = self._gcn_rnn_net(self._gcn_data_list, evaluation_set_idx)
        val_loss = self._criterion(val_output, evaluation_set_learned_labels)

        val_loss += self._l1_lambda * self._l1_norm()

        if evaluate_accuracy:
            accuracy = self._evaluate_accuracy(
                val_output, evaluation_set_learned_labels)

            tot_accuracy = self._evaluate_accuracy(
                evaluation_set_current_labels + val_output, evaluation_set_next_labels)
        else:
            accuracy = None
            tot_accuracy = None
        
        if evaluate_correlation:
            correlation = self._evaluate_correlation(
                val_output, evaluation_set_learned_labels)

            tot_correlation = self._evaluate_correlation(
                evaluation_set_current_labels + val_output, evaluation_set_next_labels)
        else:
            correlation = None
            tot_correlation = None

        return val_loss, accuracy, tot_accuracy, correlation, tot_correlation

    def train(self):
        (
            train_loss_list,
            train_accuracy_list,
            train_tot_accuracy_list,
            val_accuracy_list,
            val_loss_list,
            val_tot_accuracy_list
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for epoch in range(int(self._params["epochs"])):
            self._gcn_rnn_net.train()
            self._optimizer.zero_grad()

            train_loss, _, _, _, _ = self.evaluate("train")
            # train_loss_list.append(train_loss)
            # train_accuracy_list.append(train_accuracy)
            # train_tot_accuracy_list.append(train_tot_accuracy)
            train_loss.backward()
            self._optimizer.step()

            # Comment out to improve performance. Will run only every 100 epochs:

            # Evaluate validation set
            # self._gcn_rnn_net.eval()
            # validation_loss, validation_accuracy, validation_tot_accuracy = self.evaluate(
            #     "validation")
            # val_loss_list.append(validation_loss)
            # val_accuracy_list.append(validation_accuracy)
            # val_tot_accuracy_list.append(validation_tot_accuracy)

            if epoch % self._print_every_num_of_opochs == self._print_every_num_of_opochs - 1:
                self._gcn_rnn_net.eval()
                train_loss, train_accuracy, train_tot_accuracy, train_correlation, train_tot_correlation = self.evaluate(
                    "train", evaluate_accuracy=True, evaluate_correlation=True)
                validation_loss, validation_accuracy, validation_tot_accuracy, validation_correlation, validation_tot_correlation = self.evaluate(
                    "validation", evaluate_accuracy=True, evaluate_correlation=True)
                print(
                    f"epoch: {epoch + 1}, train loss: {train_loss.data.cpu().item():.5f}, validation loss:{validation_loss.data.cpu().item():.5f}, "
                    f"train accuracy: {train_accuracy:.5f}, validation accuracy: {validation_accuracy:.5f} "
                    f"train tot accuracy: {train_tot_accuracy:.5f}, validation tot accuracy: {validation_tot_accuracy:.5f} "
                    f"train correlation: {train_correlation:.5f}, validation correlation: {validation_correlation:.5f} "
                    f"train tot correlation: {train_tot_correlation:.5f}, validation tot correlation: {validation_tot_correlation:.5f} "
                )
        return

    def _get_stacked_labels_by_indices_set(self, labels, indices_set):
        if indices_set == "test":
            all_indices = self.test_idx
        elif indices_set == "validation":
            all_indices = self.validation_idx
        elif indices_set == "train":
            all_indices = self.train_idx
        elif indices_set == "testval":
            all_indices = set()
            all_indices = all_indices.union(
                self.test_idx).union(self.validation_idx)
        else:
            all_indices = set()
            all_indices = all_indices.union(self.train_idx).union(
                self.test_idx).union(self.validation_idx)
        stacked_labels = torch.stack(
            list(map(lambda x: self._get_labels_by_indices(x, self.test_idx), labels)))
        return stacked_labels

    def _calc_criterion(self, predictions, true_labels, criterion):
        predictions_np = predictions.cpu().detach().numpy()
        true_labels_np = true_labels.cpu().detach().numpy()
        if criterion == 'accuracy':
            return self._accruacy_metric(predictions_np, true_labels_np)
        if criterion == 'correlation':
            return pearsonr(predictions_np, true_labels_np)[0]

    def _evaluate_accuracy(self, predictions, true_labels):
        return self._calc_criterion(predictions, true_labels, 'accuracy')

    def _evaluate_correlation(self, predictions, true_labels):
        return self._calc_criterion(predictions, true_labels, 'correlation')

    def _evaluate_log_accuracy(self, predictions, true_labels):
        predictions_log = torch.log(predictions)
        true_labels_log = torch.log(true_labels)
        return self._evaluate_accuracy(predictions_log, true_labels_log)

    def evaluate_zero_model_total_num(self, labels, indices):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1])
            predictions.append(l)

            losses.append(self._criterion(predictions[-1], true_labels[-1]))
            accuracies.append(self._evaluate_accuracy(
                predictions[-1], true_labels[-1]))
            correlations.append(self._evaluate_correlation(
                predictions[-1], true_labels[-1]))
        return losses, accuracies, correlations

    def evaluate_zero_model_diff(self, labels, indices):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1]-l)
            predictions.append(torch.zeros(len(l), device=DEVICE))

            losses.append(self._criterion(predictions[-1], true_labels[-1]))
            accuracies.append(self._evaluate_accuracy(
                predictions[-1], true_labels[-1]))
            correlations.append(self._evaluate_correlation(
                predictions[-1], true_labels[-1]))
        return losses, accuracies, correlations

    def evaluate_first_order_model_total_number(self, labels, indices):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        lin_reg_models = []
        time_steps = np.arange(len(stacked_labels)-1).reshape(-1, 1)
        for idx, n in enumerate(stacked_labels.T):
            lin_reg_models.append(
                LinearRegression().fit(time_steps, n[:-1].cpu()))

        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1])
            predictions.append(torch.tensor(
                [m.predict(np.array([[idx+1]])) for m in lin_reg_models], device=DEVICE).view(-1))

            losses.append(self._criterion(predictions[idx], true_labels[idx]))
            accuracies.append(self._evaluate_accuracy(
                predictions[idx], true_labels[idx]))
            correlations.append(self._evaluate_correlation(
                predictions[idx], true_labels[idx]))

        return losses, accuracies, correlations

    def evaluate_first_order_model_diff(self, labels, indices):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        lin_reg_models = []
        time_steps = np.arange(len(stacked_labels)-1).reshape(-1, 1)
        for idx, n in enumerate(stacked_labels.T):
            diffs = []
            for idx, m in enumerate(n[:-1]):
                diffs.append(n[idx+1]-m)
            lin_reg_models.append(
                LinearRegression().fit(time_steps, diffs))

        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1]-l)
            predictions.append(torch.tensor([m.predict(
                np.array([[idx+1]])) for m in lin_reg_models], device=DEVICE).view(-1))

            losses.append(self._criterion(predictions[idx], true_labels[idx]))
            accuracies.append(self._evaluate_accuracy(
                predictions[idx], true_labels[idx]))
            correlations.append(self._evaluate_correlation(
                predictions[idx], true_labels[idx]))

        return losses, accuracies, correlations
