import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from utils.models import comparison_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphSeriesData():
    def __init__(self, log_guard_scale):
        self._device = DEVICE
        self._train_idx = None
        self._test_idx = None
        self._validation_idx = None
        self._graph_series_data_list = None
        self._learned_label = None
        self._num_of_nodes = 0
        self._node_id_to_idx = {}
        self._idx_to_node_id = {}
        self._log_guard_scale = log_guard_scale

    def load_data(
        self,
        gnxs: list,
        graph_features,
        user_node_id_to_idx: dict = None,
        user_idx_to_node_id: dict = None,
        learned_label: str = "",
        labels_list: list = None,
        learn_logs: bool = False,
        learn_diffs: bool = True,
        train=None,
        test=None,
        validation=None,
    ):
        self._graph_series_data_list = []

        self._learned_label = learned_label

        self._learn_logs = learn_logs
        self._learn_diffs = learn_diffs

        all_nodes_set = set()
        for gnx in gnxs:
            all_nodes_set.update(gnx.nodes())
        self._all_nodes_list = sorted(all_nodes_set)

        self._num_of_nodes = len(self._all_nodes_list)

        self._node_id_to_idx = {x: i for i,
                                x in enumerate(self._all_nodes_list)}
        self._idx_to_node_id = {i: x for i,
                                x in enumerate(self._all_nodes_list)}

        feature_matrices = self._get_features_by_indices(
            graph_features,
            self._all_nodes_list
        )
        self._number_of_features = feature_matrices[0].shape[1]

        for gnx, feature_matrix in zip(gnxs, feature_matrices):
            # x = torch.vstack(
            #     [
            #         feature_matrix[self._node_id_to_idx[node_id], :] for node_id in gnx.nodes()
            #     ]
            # )
            x = feature_matrix

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

            edges, _ = add_self_loops(edges, num_nodes=self._num_of_nodes)

            d = Data(x=x, edge_index=edges)

            self._graph_series_data_list.append(d)

        self._train_idx = [self._node_id_to_idx[node] for node in train]
        self._test_idx = [self._node_id_to_idx[node] for node in test]
        self._validation_idx = [self._node_id_to_idx[node]
                                for node in validation]

        if self._learn_logs:
            self._calc_log_eps(labels_list[-2:])

        self._train_next_time_labels = self._get_labels_by_indices(
            labels_list[-1], self._train_idx)
        self._train_current_labels = self._get_labels_by_indices(
            labels_list[-2], self._train_idx)

        self._test_next_time_labels = self._get_labels_by_indices(
            labels_list[-1], self._test_idx)
        self._test_current_labels = self._get_labels_by_indices(
            labels_list[-2], self._test_idx)

        self._validation_next_time_labels = self._get_labels_by_indices(
            labels_list[-1], self._validation_idx)
        self._validation_current_labels = self._get_labels_by_indices(
            labels_list[-2], self._validation_idx)

        # Special case where learning degree or rank, learn the difference between next and current time ranks.
        if self._learn_diffs:
            self._train_labels_to_learn = self._train_next_time_labels - self._train_current_labels
            self._test_labels_to_learn = self._test_next_time_labels - self._test_current_labels
            self._validation_labels_to_learn = self._validation_next_time_labels - \
                self._validation_current_labels

        else:
            self._train_labels_to_learn = self._train_next_time_labels
            self._test_labels_to_learn = self._test_next_time_labels
            self._validation_labels_to_learn = self._validation_next_time_labels

        self._train_idx = [self._node_id_to_idx[node] for node in train]
        self._test_idx = [self._node_id_to_idx[node] for node in test]
        self._validation_idx = [self._node_id_to_idx[node]
                                for node in validation]

    def _calc_log_eps(self, labels_list):
        all_indices = self._train_idx + self._test_idx + self._validation_idx
        ll = list(map(lambda x: self._get_values_by_indices(
            x, self._learned_label, all_indices), labels_list))
        assert(np.all([labels[labels > 0].shape[0] != 0 for labels in ll]))
        min_positive_label = np.min(
            [labels[labels > 0].min() for labels in ll])
        self._eps = min_positive_label/self._log_guard_scale
        return

    def _get_values_by_indices(self, values, key, indices):
        ret_values = torch.tensor(
            [values[key][self._idx_to_node_id[i]] for i in indices],
            dtype=torch.float,
            device=self._device,
        )
        return ret_values

    def _get_features_by_indices(self, features, node_idxs):
        indices = [self._node_id_to_idx[node_id] for node_id in node_idxs]
        ret_features = []
        for time_step_features in features:
            all_features = []
            for key in time_step_features.keys():
                new_features = self._get_values_by_indices(
                    time_step_features,
                    key,
                    indices
                )
                if key == 'general':
                    new_features = GraphSeriesData._sum_general_label(
                        new_features
                    )
                all_features.append(new_features.view(-1, 1))
            ret_features.append(torch.cat(all_features, dim=1))
        return ret_features

    @staticmethod
    def _sum_general_label(labels):
        if labels.dim() == 2:
            return labels.sum(dim=1)
        else:
            return labels

    def _get_labels_by_indices(self, labels, indices):
        ret_labels = self._get_values_by_indices(
            labels, self._learned_label, indices)
        # Special case where rank is required, calculate log(rank)
        if self._learned_label == "general":
            # When in_deg and out_deg are calculated seperately, learn their sum.
            ret_labels = GraphSeriesData._sum_general_label(ret_labels)
        if self._learn_logs:
            ret_labels = torch.log(ret_labels+self._eps)
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
            list(map(lambda x: self._get_labels_by_indices(x, all_indices), labels)))
        return stacked_labels

    def _calc_criterion_np(self, predictions_np, true_labels_np, criterion, loss_criterion=mean_squared_error):
        if criterion == 'r2_score':
            return r2_score(true_labels_np, predictions_np)
        if criterion == 'correlation':
            return spearmanr(true_labels_np, predictions_np).correlation
        if criterion == 'mae':
            return mean_absolute_error(true_labels_np, predictions_np)
        if criterion == 'mse':
            return mean_squared_error(true_labels_np, predictions_np)
        if criterion == 'loss':
            return loss_criterion(true_labels_np, predictions_np)

    def _prepare_and_calc_criterion(self, predictions, true_labels, criterion):
        if self._learn_logs:
            predictions = torch.exp(predictions)
            true_labels = torch.exp(true_labels)
        predictions_np = predictions.cpu().detach().numpy()
        true_labels_np = true_labels.cpu().detach().numpy()
        return self._calc_criterion_np(predictions_np, true_labels_np, criterion)

    def evaluate_r2_score(self, predictions, true_labels):
        return self._prepare_and_calc_criterion(predictions, true_labels, 'r2_score')

    def evaluate_correlation(self, predictions, true_labels):
        return self._prepare_and_calc_criterion(predictions, true_labels, 'correlation')

    def evaluate_mae(self, predictions, true_labels):
        return self._prepare_and_calc_criterion(predictions, true_labels, 'mae')

    def evaluate_mse(self, predictions, true_labels):
        return self._prepare_and_calc_criterion(predictions, true_labels, 'mse')

    def _evaluate_log_r2_score(self, predictions, true_labels):
        predictions_log = torch.log(predictions)
        true_labels_log = torch.log(true_labels)
        return self.evaluate_r2_score(predictions_log, true_labels_log)

    def _evaluate_any_model(self, model, labels, indices, loss_criterion, minimum_supported_index):
        losses = []
        accuracies = []
        correlations = []
        maes = []
        mses = []
        stacked_labels = self._get_stacked_labels_by_indices_set(
            labels,
            indices
        )
        model.train(stacked_labels.cpu().numpy())
        for t in range(minimum_supported_index, stacked_labels.shape[0]):
            model_results = model.predict_label(t)
            losses.append(
                self._calc_criterion_np(
                    model_results.model_prediction,
                    model_results.true_label,
                    "loss",
                    loss_criterion=loss_criterion
                )
            )
            accuracies.append(
                self._calc_criterion_np(
                    model_results.model_prediction,
                    model_results.true_label,
                    "r2_score"
                )
            )
            correlations.append(
                self._calc_criterion_np(
                    model_results.model_prediction,
                    model_results.true_label,
                    "correlation"
                )
            )
            maes.append(
                self._calc_criterion_np(
                    model_results.model_prediction,
                    model_results.true_label,
                    "mae"
                )
            )
            mses.append(
                self._calc_criterion_np(
                    model_results.model_prediction,
                    model_results.true_label,
                    "mse"
                )
            )
        return losses, accuracies, correlations, maes, mses

    def evaluate_null_model_total_num(self, labels, indices, loss_criterion=mean_squared_error):
        nm = comparison_models.NullModel()
        return self._evaluate_any_model(nm, labels, indices, loss_criterion, 1)

    def evaluate_null_model_diff(self, labels, indices, loss_criterion=mean_squared_error):
        nmd = comparison_models.NullDiffModel()
        return self._evaluate_any_model(nmd, labels, indices, loss_criterion, 1)

    def evaluate_first_order_total_num(self, labels, indices, average_time=None, epsilon=0, loss_criterion=mean_squared_error):
        return self.evaluate_polynomial_regression(
            labels,
            indices,
            average_time=average_time,
            degree=1,
            epsilon=epsilon,
            loss_criterion=loss_criterion
        )

    def evaluate_uniform_average(self, labels, indices, average_time=None, loss_criterion=mean_squared_error):
        if average_time is None:
            average_time = len(labels) - 1
        uam = comparison_models.UniformAverageModel(average_time)
        return self._evaluate_any_model(uam, labels, indices, loss_criterion, average_time)

    def evaluate_linear_weighted_average(self, labels, indices, average_time=None, loss_criterion=mean_squared_error):
        if average_time is None:
            average_time = len(labels) - 1
        lwam = comparison_models.LinearWeightedAverageModel(average_time)
        return self._evaluate_any_model(lwam, labels, indices, loss_criterion, average_time)

    def evaluate_square_root_weighted_average(self, labels, indices, average_time=None, loss_criterion=mean_squared_error):
        if average_time is None:
            average_time = len(labels) - 1
        srwam = comparison_models.SquareRootWeightedAverageModel(average_time)
        return self._evaluate_any_model(srwam, labels, indices, loss_criterion, average_time)

    def evaluate_polynomial_regression(self, labels, indices, average_time=None, degree=3, epsilon=0.05, loss_criterion=mean_squared_error):
        if average_time is None:
            average_time = len(labels) - 1
        if average_time < degree + 1:
            return ([np.nan],) * 5
        prm = comparison_models.PolynomialRegressionModel(
            average_time, degree, epsilon)
        return self._evaluate_any_model(prm, labels, indices, loss_criterion, average_time)

    def evaluate_uniform_periodic_average(self, labels, indices, average_time=None, period=7, epsilon=0.05, loss_criterion=mean_squared_error):
        if average_time is None:
            average_time = len(labels) - 1
        upam = comparison_models.UniformPeriodicAverageModel(
            average_time, period, epsilon)
        return self._evaluate_any_model(upam, labels, indices, loss_criterion, average_time)

    def evaluate_weighted_periodic_average(self, labels, indices, average_time=None, period=7, epsilon=0.05, loss_criterion=mean_squared_error):
        if average_time is None:
            average_time = len(labels) - 1
        wpam = comparison_models.WeightedPeriodicAverageModel(
            average_time, period, epsilon)
        return self._evaluate_any_model(wpam, labels, indices, loss_criterion, average_time)
