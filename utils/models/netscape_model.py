import torch
from torch import nn, optim
from abc import ABCMeta, abstractmethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NETSCAPEModel(metaclass=ABCMeta):
    def __init__(self, parameters, graph_series_data):
        self._params = parameters
        self._graph_series_data = graph_series_data
        self._device = DEVICE
        self._criterion = nn.MSELoss()
        self._net = None
        self._optimizer = None
        self._print_every_num_of_epochs = parameters['print_every_num_of_epochs']

        self._net = self._get_internal_net()

        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=self._params["weight_decay"],
        )

        self._l1_lambda = self._params["l1_lambda"]

        print(
            f"number of params in model: {sum(p.numel() for p in self._net.parameters())}")

        return

    @abstractmethod
    def _get_internal_net(self):
        pass

    def _ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def _l1_norm(self):
        l1_norm = 0
        for p in self._net.parameters():
            l1_norm += p.abs().sum()
        return l1_norm

    def _get_evaluation_set(self, evaluation_set: str = "test"):
        if evaluation_set == "test":
            self._net.eval()
            (
                evaluation_set_idx,
                evaluation_set_learned_labels,
                evaluation_set_current_labels,
                evaluation_set_next_labels
            ) = self._graph_series_data.get_test_data()
        elif evaluation_set == "validation":
            self._net.eval()
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
            self._net.eval()
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

    @abstractmethod
    def _evaluate_secondary_metric(
            self,
            secondary_metric,
            should_evaluate,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
    ):
        pass

    def evaluate(self, evaluation_set: str = "test", evaluate_accuracy: bool = False, evaluate_correlation: bool = False, evaluate_mae: bool = False, evaluate_mse: bool = False):

        (
            evaluation_set_idx,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        ) = self._get_evaluation_set(evaluation_set)

        val_output = self._net(
            self._graph_series_data.get_gcn_data(), evaluation_set_idx)
        val_loss = self._criterion(val_output, evaluation_set_learned_labels)

        val_loss += self._l1_lambda * self._l1_norm()

        accuracy = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_r2_score,
            evaluate_accuracy,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        correlation = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_correlation,
            evaluate_correlation,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        mae = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_mae,
            evaluate_mae,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        mse = self._evaluate_secondary_metric(
            self._graph_series_data.evaluate_mse,
            evaluate_mse,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
        )

        return val_output, val_loss, accuracy, correlation, mae, mse

    def train(self):

        for epoch in range(int(self._params["epochs"])):
            self._net.train()
            self._optimizer.zero_grad()

            _, train_loss, _, _, _, _ = self.evaluate("train")
            train_loss.backward()
            self._optimizer.step()

            if epoch % self._print_every_num_of_epochs == self._print_every_num_of_epochs - 1:
                self._net.eval()
                (
                    _,
                    train_loss,
                    train_accuracy,
                    train_correlation,
                    train_mae,
                    train_mse
                ) = self.evaluate(
                    "train",
                    evaluate_accuracy=True,
                    evaluate_correlation=True,
                    evaluate_mae=True,
                    evaluate_mse=True
                )
                (
                    _,
                    validation_loss,
                    validation_accuracy,
                    validation_correlation,
                    validation_mae,
                    validation_mse
                ) = self.evaluate(
                    "validation",
                    evaluate_accuracy=True,
                    evaluate_correlation=True,
                    evaluate_mae=True,
                    evaluate_mse=True
                )
                print(
                    f"epoch: {epoch + 1}, train loss: {train_loss.data.cpu().item():.5f}, validation loss:{validation_loss.data.cpu().item():.5f}, "
                    f"train accuracy: {train_accuracy:.5f}, validation accuracy: {validation_accuracy:.5f} "
                    f"train correlation: {train_correlation:.5f}, validation correlation: {validation_correlation:.5f} "
                    f"train mean absolute error: {train_mae:.5f}, validation mean absolute error: {validation_mae:.5f} "
                    f"train mean squared error: {train_mse:.5f}, validation mean squared error: {validation_mse:.5f} "
                )
        return
