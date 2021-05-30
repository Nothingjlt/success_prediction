import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphSeriesData():
    def __init__(self):
        self._device = DEVICE
        self._train_idx = None
        self._test_idx = None
        self._validation_idx = None
        self._graph_series_data_list = None
        self._learned_label = None
        self._num_of_nodes = 0
        self._node_id_to_idx = {}
        self._idx_to_node_id = {}

    def load_data(
        self,
        gnxs: list,
        feature_matrix,
        learned_label: str = "",
        current_timestamp_labels=None,
        next_timestamp_labels=None,
        train=None,
        test=None,
        validation=None,
    ):
        self._graph_series_data_list = []

        self._number_of_features = feature_matrix.shape[1]

        self._learned_label = learned_label

        all_nodes_set = set()
        for gnx in gnxs:
            all_nodes_set.update(gnx.nodes())
        self._all_nodes_list = sorted(all_nodes_set)

        self._num_of_nodes = len(self._all_nodes_list)

        self._node_id_to_idx = {x: i for i,
                                x in enumerate(self._all_nodes_list)}
        self._idx_to_node_id = {i: x for i,
                                x in enumerate(self._all_nodes_list)}

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

            self._graph_series_data_list.append(d)

        self._train_idx = [self._node_id_to_idx[node] for node in train]
        self._test_idx = [self._node_id_to_idx[node] for node in test]
        self._validation_idx = [self._node_id_to_idx[node]
                               for node in validation]

        self._train_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self._train_idx)
        self._train_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self._train_idx)

        self._test_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self._test_idx)
        self._test_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self._test_idx)

        self._validation_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self._validation_idx)
        self._validation_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self._validation_idx)

        # Special case where both in_deg and out_deg are learned, reduce to deg.
        if self._learned_label == "general":

            # Trials with Yoram
            self._train_labels_to_learn = self._train_next_time_labels - self._train_current_labels
            self._test_labels_to_learn = self._test_next_time_labels - self._test_current_labels
            self._validation_labels_to_learn = self._validation_next_time_labels - \
                self._validation_current_labels

            # End trials with Yoram
            # self._train_next_time_labels = self._train_next_time_labels.sum(
            #     dim=1).log()
            # self._test_next_time_labels = self._test_next_time_labels.sum(
            #     dim=1).log()
            # self._validation_next_time_labels = self._validation_next_time_labels.sum(
            #     dim=1).log()

        self._train_idx = [self._node_id_to_idx[node] for node in train]
        self._test_idx = [self._node_id_to_idx[node] for node in test]
        self._validation_idx = [self._node_id_to_idx[node]
                               for node in validation]
    
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
        return self._graph_series_data_list.clone()
    
    def get_number_of_features(self):
        return self._number_of_features
    
    def get_gcn_data(self):
        return self._graph_series_data_list
    
    def get_train_data(self):
        return self._train_idx, self._train_labels_to_learn, self._train_current_labels, self._train_next_time_labels

    def get_test_data(self):
        return self._test_idx, self._test_labels_to_learn, self._test_current_labels, self._test_next_time_labels

    def get_validation_data(self):
        return self._validation_idx, self._validation_labels_to_learn, self._validation_current_labels, self._validation_next_time_labels

    def _get_stacked_labels_by_indices_set(self, labels, indices_set):
        if indices_set == "test":
            all_indices = self._test_idx
        elif indices_set == "validation":
            all_indices = self._validation_idx
        elif indices_set == "train":
            all_indices = self._train_idx
        elif indices_set == "testval":
            all_indices = set()
            all_indices = all_indices.union(
                self._test_idx).union(self._validation_idx)
        else:
            all_indices = set()
            all_indices = all_indices.union(self._train_idx).union(
                self._test_idx).union(self._validation_idx)
        stacked_labels = torch.stack(
            list(map(lambda x: self._get_labels_by_indices(x, self._test_idx), labels)))
        return stacked_labels

    def _calc_criterion(self, predictions, true_labels, criterion):
        predictions_np = predictions.cpu().detach().numpy()
        true_labels_np = true_labels.cpu().detach().numpy()
        if criterion == 'r2_score':
            return r2_score(predictions_np, true_labels_np)
        if criterion == 'correlation':
            return pearsonr(predictions_np, true_labels_np)[0]
        if criterion == 'mae':
            return mean_absolute_error(predictions_np, true_labels_np)
    
    def evaluate_r2_score(self, predictions, true_labels):
        return self._calc_criterion(predictions, true_labels, 'r2_score')

    def evaluate_correlation(self, predictions, true_labels):
        return self._calc_criterion(predictions, true_labels, 'correlation')

    def evaluate_mae(self, predictions, true_labels):
        return self._calc_criterion(predictions, true_labels, 'mae')

    def _evaluate_log_r2_score(self, predictions, true_labels):
        predictions_log = torch.log(predictions)
        true_labels_log = torch.log(true_labels)
        return self.evaluate_r2_score(predictions_log, true_labels_log)
    
    def evaluate_zero_model_total_num(self, labels, indices, loss_criterion=torch.nn.MSELoss()):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        maes = []
        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1])
            predictions.append(l)

            losses.append(loss_criterion(predictions[-1], true_labels[-1]))
            accuracies.append(self.evaluate_r2_score(
                predictions[-1], true_labels[-1]))
            correlations.append(self.evaluate_correlation(
                predictions[-1], true_labels[-1]))
            maes.append(self.evaluate_mae(
                predictions[-1], true_labels[-1]))
        return losses, accuracies, correlations, maes

    def evaluate_zero_model_diff(self, labels, indices, loss_criterion=torch.nn.MSELoss()):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        maes = []
        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1]-l)
            predictions.append(torch.zeros(len(l), device=DEVICE))

            losses.append(loss_criterion(predictions[-1], true_labels[-1]))
            accuracies.append(self.evaluate_r2_score(
                predictions[-1], true_labels[-1]))
            correlations.append(self.evaluate_correlation(
                predictions[-1], true_labels[-1]))
            maes.append(self.evaluate_mae(
                predictions[-1], true_labels[-1]))
        return losses, accuracies, correlations, maes

    def evaluate_first_order_model_total_number(self, labels, indices, loss_criterion=torch.nn.MSELoss()):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        maes = []
        lin_reg_models = []
        time_steps = np.arange(len(stacked_labels)-1).reshape(-1, 1)
        for idx, n in enumerate(stacked_labels.T):
            lin_reg_models.append(
                LinearRegression().fit(time_steps, n[:-1].cpu()))

        for idx, l in enumerate(stacked_labels[:-1]):
            true_labels.append(stacked_labels[idx + 1])
            predictions.append(torch.tensor(
                [m.predict(np.array([[idx+1]])) for m in lin_reg_models], device=DEVICE).view(-1))

            losses.append(loss_criterion(predictions[idx], true_labels[idx]))
            accuracies.append(self.evaluate_r2_score(
                predictions[idx], true_labels[idx]))
            correlations.append(self.evaluate_correlation(
                predictions[idx], true_labels[idx]))
            maes.append(self.evaluate_mae(
                predictions[idx], true_labels[idx]))

        return losses, accuracies, correlations, maes

    def evaluate_first_order_model_diff(self, labels, indices, loss_criterion=torch.nn.MSELoss()):
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels, indices)
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        correlations = []
        maes = []
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

            losses.append(loss_criterion(predictions[idx], true_labels[idx]))
            accuracies.append(self.evaluate_r2_score(
                predictions[idx], true_labels[idx]))
            correlations.append(self.evaluate_correlation(
                predictions[idx], true_labels[idx]))
            maes.append(self.evaluate_mae(
                predictions[idx], true_labels[idx]))

        return losses, accuracies, correlations, maes