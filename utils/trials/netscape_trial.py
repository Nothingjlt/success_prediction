import torch
import pickle
import numpy as np
from lib.graph_measures.features_meta.features_meta import *
from utils.utils import GraphSeriesData
from utils.general.trial_summary import TrialSummary
from abc import ABCMeta, abstractmethod

import argparse

import nni


class NETSCAPETrial(metaclass=ABCMeta):

    def __init__(self, parameters: dict = {}, seed: int = None, out_folder: str = "out"):
        seed_to_set = np.random.randint(0, 2 ^ 32) if seed is None else seed
        self._set_seed(seed_to_set)
        self._params = parameters
        self._default_features_meta = {
            "betweenness_centrality": FeatureMeta(
                BetweennessCentralityCalculator, {"betweenness"}
            ),
            "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
            "k_core": FeatureMeta(KCoreCalculator, {"kcore"}),
            "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load"}),
            "page_rank": FeatureMeta(PageRankCalculator, {"page"}),
            "general": FeatureMeta(GeneralCalculator, {"gen"}),
        }
        self._all_labels_together = "all_labels"
        self._undirected_features_names_to_lengths = {
            'average_neighbor_degree': 1,
            'eccentricity': 1,
            'fiedler_vector': 1,
            'louvain': 1,
            'motif3': 2,
            'motif4': 6,
            'eigenvector_centrality': 1,
            'clustering_coefficient': 1,
            'square_clustering_coefficient': 1,
            # actually a list of length 1 containing a tuple of length 2
            'bfs_moments': [1, 2]
        }
        self._directed_features_names_to_lengths = {
            # 'attractor_basin': 1, # attractor_basin can be nan for nodes with no in edges or no out edges
            'average_neighbor_degree': 1,
            'eccentricity': 1,
            'flow': 1,
            'motif3': 13,
            'motif4': 199,
            'eigenvector_centrality': 1,
            'clustering_coefficient': 1,
            'square_clustering_coefficient': 1,
            # actually a list of length 1 containing a tuple of length 2
            'bfs_moments': [1, 2]
        }

        self._default_out_dir = out_folder
        self._nni = False
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self, seed):
        self._seed = seed
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        import random
        random.seed = self._seed
        import os
        os.environ['PYTHONHASHSEED'] = str(self._seed)

    @staticmethod
    def train_test_split(graphs, train_ratio):
        prev_graph = graphs[0]
        all_intersection = np.array(prev_graph.nodes())
        for g in graphs[1:]:
            all_intersection = np.intersect1d(all_intersection, g.nodes())
            prev_graph = g

        print("# of nodes that appear at all timestamps", len(all_intersection))
        val_test_inds = np.random.choice(
            all_intersection,
            round(
                len(all_intersection) * (1-train_ratio)
            ),
            replace=False
        )
        test, validation = np.array_split(val_test_inds, 2)
        train = np.setdiff1d(np.array(all_intersection), val_test_inds)
        e = 0
        return train, test, validation

    def _get_inupt_pickle_file_name(self):
        input_pickle_file_name = "./Pickles/" + str(self._params["data_folder_name"]) + \
            "/" + self._params["data_name"] + \
            "_with_features" + ".pkl"
        return input_pickle_file_name

    def load_input(self):
        input_pickle_file_name = self._get_inupt_pickle_file_name()
        graphs, all_features = pickle.load(open(input_pickle_file_name, "rb"))
        all_nodes = set()
        for g in graphs:
            all_nodes.update(g.nodes())

        self._directed = g.is_directed()

        self._all_nodes_list = sorted(all_nodes)

        self._node_id_to_idx = {
            x: i for i,
            x in enumerate(self._all_nodes_list)
        }
        self._idx_to_node_id = {
            i: x for i,
            x in enumerate(self._all_nodes_list)
        }

        labels = self._get_labels(all_features)
        graph_features = self._get_features(all_features)

        return graphs, labels, graph_features

    def _add_learned_label_to_features(self, labels, graph_features, add_labels_of_all_times: bool = False):
        assert len(labels) == len(graph_features)
        for label, graph_feature in zip(labels, graph_features):
            if add_labels_of_all_times:
                graph_feature[
                    self._params["learned_label"]
                ] = label[self._params["learned_label"]]
            # Add label of last time step that is not the predicted one
            else:
                graph_feature[
                    self._params["learned_label"]
                ] = labels[-2][self._params["learned_label"]]
        return graph_features

    def run_trial(self):
        print(self._params)
        learned_label = self._params["learned_label"]
        graphs_orig, labels_orig, graph_features_orig = self.load_input()
        graph_features_orig = self._add_learned_label_to_features(
            labels_orig,
            graph_features_orig,
            self._params["add_labels_of_all_times"]
        )
        # GPU memory limited, using only subset of time steps
        graphs, labels, graph_features = self._cut_graphs_list(
            graphs_orig,
            labels_orig,
            graph_features_orig
        )
        train, test, validation = NETSCAPETrial.train_test_split(
            graphs,
            train_ratio=0.7
        )

        graph_data = GraphSeriesData(self._params["log_guard_scale"])

        graph_data.load_data(
            graphs[:-1],
            graph_features[:-1],
            user_node_id_to_idx=self._node_id_to_idx,
            user_idx_to_node_id=self._idx_to_node_id,
            learned_label=learned_label,
            labels_list=labels[-2:],
            learn_logs=self._should_learn_logs(),
            learn_diffs=self._should_learn_diff(),
            train=train,
            test=test,
            validation=validation
        )

        model = self._get_model(graph_data)

        self._set_seed(self._seed)

        def weight_reset(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
                # for k, v in m._parameters.items():
                #     print(k, v)
                return
        model._net.apply(weight_reset)

        model.train()
        with torch.no_grad():
            model_test_evaluation = TrialSummary.model_evaluation_results_type(
                *model.evaluate(
                    "test",
                    evaluate_accuracy=True,
                    evaluate_correlation=True,
                    evaluate_mae=True,
                    evaluate_mse=True
                )[1:]
            )
            model_test_evaluation = model_test_evaluation._replace(
                losses=[model_test_evaluation.losses.cpu().numpy()],
                accuracies=[model_test_evaluation.accuracies],
                correlations=[model_test_evaluation.correlations],
                maes=[model_test_evaluation.maes],
                mses=[model_test_evaluation.mses]
            )

            model_train_evaluation = TrialSummary.model_evaluation_results_type(
                *model.evaluate(
                    "train",
                    evaluate_accuracy=True,
                    evaluate_correlation=True,
                    evaluate_mae=True,
                    evaluate_mse=True
                )[1:]
            )
            model_train_evaluation = model_train_evaluation._replace(
                losses=[model_train_evaluation.losses.cpu().numpy()],
                accuracies=[model_train_evaluation.accuracies],
                correlations=[model_train_evaluation.correlations],
                maes=[model_train_evaluation.maes],
                mses=[model_train_evaluation.mses]
            )

            comparison_graph_data = GraphSeriesData(log_guard_scale=0)
            comparison_graph_data.load_data(
                graphs_orig,
                graph_features_orig,
                user_node_id_to_idx=self._node_id_to_idx,
                user_idx_to_node_id=self._idx_to_node_id,
                learned_label=learned_label,
                labels_list=labels_orig,
                learn_logs=False,
                learn_diffs=False,
                train=train,
                test=test,
                validation=validation
            )

            null_model_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_null_model_total_num(labels_orig, "test")
            )
            first_order_model_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_first_order_total_num(labels_orig, "test")
            )
            # null_model_diff_evaluation = TrialSummary.model_evaluation_results_type(
            #     *comparison_graph_data.evaluate_null_model_diff(labels_orig, "test")
            # )
            null_model_diff_evaluation = TrialSummary.model_evaluation_results_type(
                [np.inf], [-np.inf], [0], [np.inf], [np.inf])
            uniform_average_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_uniform_average(labels_orig, "test")
            )
            linear_weighted_average_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_linear_weighted_average(labels_orig, "test")
            )
            square_root_weighted_average_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_square_root_weighted_average(labels_orig, "test")
            )
            polynomial_regression_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_polynomial_regression(labels_orig, "test")
            )
            uniform_periodic_average_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_uniform_periodic_average(labels_orig, "test")
            )
            weighted_periodic_average_evaluation = TrialSummary.model_evaluation_results_type(
                *comparison_graph_data.evaluate_weighted_periodic_average(labels_orig, "test")
            )

        if self._nni:
            nni.report_intermediate_result(
                model_test_evaluation.accuracies -
                null_model_evaluation.accuracies[-1]
            )

        all_results = TrialSummary.all_results_type(
            model_test_evaluation,
            model_train_evaluation,
            null_model_evaluation,
            first_order_model_evaluation,
            null_model_diff_evaluation,
            uniform_average_evaluation,
            linear_weighted_average_evaluation,
            square_root_weighted_average_evaluation,
            polynomial_regression_evaluation,
            uniform_periodic_average_evaluation,
            weighted_periodic_average_evaluation
        )

        for name, evaluation in all_results._asdict().items():
            print(
                f"{name.replace('_', ' ')} mse: {np.mean([m for m in evaluation.mses]):.5f}, "
                f"{name.replace('_', ' ')} accuracy: {np.mean(evaluation.accuracies):.5f}, "
                f"{name.replace('_', ' ')} correlation: {np.mean(evaluation.correlations):.5f}, "
                f"{name.replace('_', ' ')} mae: {np.mean(evaluation.maes):.5f}, "
                f"{name.replace('_', ' ')} loss: {np.mean(evaluation.losses):.5f}"
            )
        print("\n")

        return all_results, TrialSummary.trial_data(model, train, test, validation)

    @abstractmethod
    def _get_model(self, graph_data):
        pass

    @abstractmethod
    def _should_learn_diff(self):
        pass

    @abstractmethod
    def _should_learn_logs(self):
        pass

    @abstractmethod
    def _cut_graphs_list(self, graphs_before_cut, labels_before_cut):
        pass

    def _get_labels(self, all_features):
        return [{k: all_feature[k] for k in self._default_features_meta.keys()} for all_feature in all_features]

    def _get_features(self, all_features):
        ret_features = []
        feautres_names_to_lengths = self._directed_features_names_to_lengths if self._directed else self._undirected_features_names_to_lengths
        for time_step_features in all_features:
            flattened_feature_dict = {}
            for feature, lengths in feautres_names_to_lengths.items():
                # make sure all features are stored in an iterable fashion
                if isinstance(lengths, int):
                    if lengths == 1:
                        flattened_feature_dict[feature] = time_step_features[feature]
                    else:
                        for i in range(lengths):
                            new_feature = feature + f"_{i}"
                            flattened_feature_dict[new_feature] = {
                                node: time_step_features[feature][node][i] for node in time_step_features[feature].keys()}
                else:  # One special case of bfs_moments
                    for i in range(lengths[1]):
                        new_feature = feature + f"_{i}"
                        flattened_feature_dict[new_feature] = {
                            node: time_step_features[feature][node][0][i] for node in time_step_features[feature].keys()}
            ret_features.append(flattened_feature_dict)
        return ret_features

    def _print_all_results(self, all_results):
        mses = {}
        accuracies = {}
        correlations = {}
        maes = {}
        for name, evaluation in all_results._asdict().items():
            mses[name] = evaluation.mses
            accuracies[name] = evaluation.accuracies
            correlations[name] = evaluation.correlations
            maes[name] = evaluation.maes
        mses_string_list = ["Final result (mse):"]
        accuracies_string_list = ["Final result (accuracy):"]
        correlations_string_list = ["Final result (correlation):"]
        maes_string_list = ["Final result (mae):"]
        for name, mse in mses.items():
            mses_string_list.append(
                f"{name.replace('_', ' ')} mse:"
            )
            mses_string_list.append(f"{mse},")
        # Remove last comma
        mses_string_list[-1] = mses_string_list[-1][:-1]
        for name, accuracy in accuracies.items():
            accuracies_string_list.append(
                f"{name.replace('_', ' ')} accuracy:"
            )
            accuracies_string_list.append(f"{accuracy},")
        # Remove last comma
        accuracies_string_list[-1] = accuracies_string_list[-1][:-1]
        for name, correlation in correlations.items():
            correlations_string_list.append(
                f"{name.replace('_', ' ')} correlation:"
            )
            correlations_string_list.append(f"{correlation},")
        # Remove last comma
        correlations_string_list[-1] = correlations_string_list[-1][:-1]
        for name, mae in maes.items():
            maes_string_list.append(f"{name.replace('_', ' ')} mae:")
            maes_string_list.append(f"{mae},")
        # Remove last comma
        maes_string_list[-1] = maes_string_list[-1][:-1]

        print(" ".join(mses_string_list))
        print(" ".join(accuracies_string_list))
        print(" ".join(correlations_string_list))
        print(" ".join(maes_string_list))

        return

    def run_one_test_iteration(self):

        all_results, model_data = self.run_trial()

        if self._nni:
            nni.report_final_result(
                all_results.model_test.accuracies[-1] -
                all_results.null_model.accuracies[-1]
            )
        else:
            self._print_all_results(all_results)

        return all_results, model_data

    def _get_output_file_name(self):
        output_file_name = "./" + self._default_out_dir + "/" + \
            str(self._params["data_folder_name"]) + "/" + \
            "_".join([self._params["learned_label"],
                      self._params["data_name"], self._get_trial_name()]) + ".out"
        return output_file_name

    def iterate_test(self):
        results = []
        models_data_list = []
        self._output_file_name = self._get_output_file_name()
        logger = TrialSummary(self._output_file_name)

        for i in range(self._params["num_iterations"]):
            print(
                f'Iteration number {i+1} out of {self._params["num_iterations"]}'
            )
            result, model_data = self.run_one_test_iteration()
            results.append(result)
            models_data_list.append(model_data)
        print('-'*100)
        for result_idx, result in enumerate(results):
            print(f"iteration {result_idx + 1} results")
            self._print_all_results(result)
        logger.write_output(results)
        logger.dump_models(models_data_list)
        return results

    def _add_general_parser_arguments(self):
        self._argparser.add_argument("--nni", action='store_true',
                                     help="Should only be used when run with nni.")
        self._argparser.add_argument("--seed", type=int,
                                     help="Optional random seed for run")
        self._argparser.add_argument("--data-folder-name", type=str, default="reality_mining",
                                     help="Folder name for dataset to evaluate")
        self._argparser.add_argument("--data-name", type=str, default="reality_mining_daily",
                                     help="Data pickle file name, without the .pkl extention")
        self._argparser.add_argument("--num-iterations", type=int, default=30,
                                     help="Number of iterations per test")
        self._argparser.add_argument("--print-every-num-of-epochs", type=int, default=100,
                                     help="Number of epochs between summary prints")
        self._argparser.add_argument("--log-guard-scale", type=float, default=10,
                                     help="Scale of log guard, used to guard against taking log of 0")
        self._argparser.add_argument("--l1-lambda", type=float, default=5e-6,
                                     help="L1 norm regularization weight")
        self._argparser.add_argument("--epochs", type=int, default=500,
                                     help="Number of epochs to learn")
        self._argparser.add_argument("--learning-rate", type=float, default=1e-3,
                                     help="Learning rate for network")
        self._argparser.add_argument("--weight-decay", type=float, default=3e-2,
                                     help="Weight decay regularization")
        self._argparser.add_argument("--add-labels-of-all-times", action='store_true',
                                     help="Decide whether learned label of each time should be added to features. "
                                     "By default only learned label of last learned time is added to all times")
        learned_labels_choices = set(self._default_features_meta.keys())
        learned_labels_choices.add(self._all_labels_together)
        self._argparser.add_argument("--learned-label", type=str, default="betweenness_centrality", choices=learned_labels_choices,
                                     help="Type of label to learn")

    def _update_general_parser_arguments(self, args):
        self._params.update(vars(args))

        if args.seed is not None:
            self._set_seed(args.seed)

        self._nni = args.nni

    @abstractmethod
    def _add_specific_parser_arguments(self):
        pass

    @abstractmethod
    def _update_specific_parser_arguments(self, args):
        pass

    @abstractmethod
    def _get_trial_name(self):
        pass

    def prepare_params(self):
        self._argparser = argparse.ArgumentParser()
        self._add_general_parser_arguments()
        self._add_specific_parser_arguments()

        args = self._argparser.parse_args()

        self._update_general_parser_arguments(args)
        self._update_specific_parser_arguments(args)

        return

    def set_learned_label(self, learned_label):
        self._params["learned_label"] = learned_label

    def run_full_trial(self):
        self.prepare_params()

        if self._params['learned_label'] == self._all_labels_together:
            for learned_label in self._default_features_meta.keys():
                self.set_learned_label(learned_label)
                self.iterate_test()
        else:
            self.iterate_test()
