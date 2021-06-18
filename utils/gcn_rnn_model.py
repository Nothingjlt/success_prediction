import torch
from torch import nn, optim
from utils.gcn_rnn_net import GCNRNNNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNRNNModel:
    def __init__(self, parameters, graph_series_data):
        self._params = parameters
        self._graph_series_data = graph_series_data
        self._device = DEVICE
        self._criterion = nn.MSELoss()
        self._gcn_rnn_net = None
        self._optimizer = None
        self._print_every_num_of_epochs = parameters['print_every_num_of_epochs']

        self._gcn_rnn_net = GCNRNNNet(
            self._graph_series_data.get_number_of_features(),
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

    def _ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def _l1_norm(self):
        l1_norm = 0
        for p in self._gcn_rnn_net.parameters():
            l1_norm += p.abs().sum()
        return l1_norm

    def _get_evaluation_set(self, evaluation_set: str = "test"):
        if evaluation_set == "test":
            self._gcn_rnn_net.eval()
            (
                evaluation_set_idx,
                evaluation_set_learned_labels,
                evaluation_set_current_labels,
                evaluation_set_next_labels
            ) = self._graph_series_data.get_test_data()
        elif evaluation_set == "validation":
            self._gcn_rnn_net.eval()
            (
                evaluation_set_idx,
                evaluation_set_learned_labels,
                evaluation_set_current_labels,
                evaluation_set_next_labels
            ) = self._graph_series_data.get_validation_data()
        elif evaluation_set == "train":
            (
                evaluation_set_idx,
                evaluation_set_learned_labels,
                evaluation_set_current_labels,
                evaluation_set_next_labels
            ) = self._graph_series_data.get_train_data()
        elif evaluation_set == "testval":
            self._gcn_rnn_net.eval()
            (
                validation_set_idx,
                validation_set_learned_labels,
                validation_set_current_labels,
                validation_set_next_labels
            ) = self._graph_series_data.get_validation_data()
            (
                test_set_idx,
                test_set_learned_labels,
                test_set_current_labels,
                test_set_next_labels
            ) = self._graph_series_data.get_test_data()
            evaluation_set_idx = validation_set_idx + test_set_idx
            evaluation_set_learned_labels = torch.cat(
                (validation_set_learned_labels, test_set_learned_labels))
            evaluation_set_current_labels = torch.cat(
                (validation_set_current_labels, test_set_current_labels))
            evaluation_set_next_labels = torch.cat(
                (validation_set_next_labels, test_set_next_labels))
        else:
            print(f"Invalid evaluation set: {evaluation_set}")
            return
        return evaluation_set_idx, evaluation_set_learned_labels, evaluation_set_current_labels, evaluation_set_next_labels

    def _evaluate_secondary_metric(
            self,
            secondary_metric,
            should_evaluate,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
    ):
        metric = None
        tot_metric = None
        if should_evaluate:
            metric = secondary_metric(
                val_output, evaluation_set_learned_labels)
            tot_metric = secondary_metric(
                evaluation_set_current_labels + val_output, evaluation_set_next_labels)

        return metric, tot_metric

    def evaluate(self, evaluation_set: str = "test", evaluate_accuracy: bool = False, evaluate_correlation: bool = False, evaluate_mae: bool = False):
        # self._gcn_rnn_net.eval()

        (
            evaluation_set_idx,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        ) = self._get_evaluation_set(evaluation_set)

        val_output = self._gcn_rnn_net(
            self._graph_series_data.get_gcn_data(), evaluation_set_idx)
        val_loss = self._criterion(val_output, evaluation_set_learned_labels)

        val_loss += self._l1_lambda * self._l1_norm()

        accuracy, tot_accuracy = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_r2_score,
            evaluate_accuracy,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        correlation, tot_correlation = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_correlation,
            evaluate_correlation,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        mae, tot_mae = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_mae,
            evaluate_mae,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        return val_loss, accuracy, tot_accuracy, correlation, tot_correlation, mae, tot_mae

    def train(self):

        for epoch in range(int(self._params["epochs"])):
            self._gcn_rnn_net.train()
            self._optimizer.zero_grad()

            train_loss, _, _, _, _, _, _ = self.evaluate("train")
            train_loss.backward()
            self._optimizer.step()

            if epoch % self._print_every_num_of_epochs == self._print_every_num_of_epochs - 1:
                self._gcn_rnn_net.eval()
                (
                    train_loss,
                    train_accuracy,
                    train_tot_accuracy,
                    train_correlation,
                    train_tot_correlation,
                    train_mae,
                    train_tot_mae
                ) = self.evaluate(
                    "train",
                    evaluate_accuracy=True,
                    evaluate_correlation=True,
                    evaluate_mae=True
                )
                (
                    validation_loss,
                    validation_accuracy,
                    validation_tot_accuracy,
                    validation_correlation,
                    validation_tot_correlation,
                    validation_mae,
                    validation_tot_mae
                ) = self.evaluate(
                    "validation",
                    evaluate_accuracy=True,
                    evaluate_correlation=True,
                    evaluate_mae=True
                )
                print(
                    f"epoch: {epoch + 1}, train loss: {train_loss.data.cpu().item():.5f}, validation loss:{validation_loss.data.cpu().item():.5f}, "
                    f"train accuracy: {train_accuracy:.5f}, validation accuracy: {validation_accuracy:.5f} "
                    f"train tot accuracy: {train_tot_accuracy:.5f}, validation tot accuracy: {validation_tot_accuracy:.5f} "
                    f"train correlation: {train_correlation:.5f}, validation correlation: {validation_correlation:.5f} "
                    f"train tot correlation: {train_tot_correlation:.5f}, validation tot correlation: {validation_tot_correlation:.5f} "
                    f"train mean absolute error: {train_mae:.5f}, validation mean absolute error: {validation_mae:.5f} "
                    f"train tot mean absolute error: {train_tot_mae:.5f}, validation tot mean absolute error: {validation_tot_mae:.5f} "
                )
        return
