import torch
import pickle
import numpy as np
from lib.graph_measures.features_meta.features_meta import *
from utils.utils import GraphSeriesData
from utils.general.trial_summary import TrialSummary

import argparse


class ComparisonTrial():
    def __init__(self, parameters: dict = {}, out_folder: str = "out"):
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

    def run_trial(self):
        print(self._params)
        learned_label = self._params["learned_label"]
        graphs, labels, graph_features = self.load_input()
        train, test, validation = ComparisonTrial.train_test_split(
            graphs,
            train_ratio=1
        )

        graph_data = GraphSeriesData(log_guard_scale=0)

        graph_data.load_data(
            graphs,
            graph_features,
            user_node_id_to_idx=self._node_id_to_idx,
            user_idx_to_node_id=self._idx_to_node_id,
            learned_label=learned_label,
            labels_list=labels,
            learn_logs=False,
            learn_diffs=False,
            train=train,
            test=test,
            validation=validation
        )

        null_model_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_null_model_total_num(labels, "train")
        )
        first_order_model_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_first_order_total_num(labels, "train")
        )
        # null_model_diff_evaluation = TrialSummary.model_evaluation_results_type(
        #     *graph_data.evaluate_null_model_diff(labels, "train")
        # )
        null_model_diff_evaluation = TrialSummary.model_evaluation_results_type([0], [0], [0], [0])
        uniform_average_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_uniform_average(labels, "train")
        )
        linear_weighted_average_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_linear_weighted_average(labels, "train")
        )
        square_root_weighted_average_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_square_root_weighted_average(labels, "train")
        )
        polynomial_regression_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_polynomial_regression(labels, "train")
        )
        uniform_periodic_average_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_uniform_periodic_average(labels, "train")
        )
        weighted_periodic_average_evaluation = TrialSummary.model_evaluation_results_type(
            *graph_data.evaluate_weighted_periodic_average(labels, "train")
        )

        all_results = TrialSummary.all_results_type(
            TrialSummary.model_evaluation_results_type([0], [0], [0], [0]),
            TrialSummary.model_evaluation_results_type([0], [0], [0], [0]),
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
                f"{name.replace('_', ' ')} mae: {np.mean(evaluation.maes):.5f}"
            )
        print("\n")

        return all_results

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

        all_results = self.run_trial()

        self._print_all_results(all_results)

        return all_results

    def _get_output_file_name(self):
        output_file_name = "./" + self._default_out_dir + "/" + \
            str(self._params["data_folder_name"]) + "/" + \
            "_".join([self._params["learned_label"],
                      self._params["data_name"], "comparison_data"]) + ".out"
        return output_file_name

    def iterate_test(self):
        results = []
        output_file_name = self._get_output_file_name()
        logger = TrialSummary(output_file_name)

        results.append(self.run_one_test_iteration())
        print('-'*100)
        for result_idx, result in enumerate(results):
            print(f"iteration {result_idx + 1} results")
            self._print_all_results(result)
        logger.write_output(results)
        return results

    def _add_general_parser_arguments(self):
        self._argparser.add_argument("--data-folder-name", type=str, default="reality_mining",
                                     help="Folder name for dataset to evaluate")
        self._argparser.add_argument("--data-name", type=str, default="reality_mining_daily",
                                     help="Data pickle file name, without the .pkl extention")
        learned_labels_choices = set(self._default_features_meta.keys())
        learned_labels_choices.add(self._all_labels_together)
        self._argparser.add_argument("--learned-label", type=str, default="betweenness_centrality", choices=learned_labels_choices,
                                     help="Type of label to learn")

    def _update_general_parser_arguments(self, args):
        self._params.update(vars(args))

    def prepare_params(self):
        self._argparser = argparse.ArgumentParser()
        self._add_general_parser_arguments()

        args = self._argparser.parse_args()

        self._update_general_parser_arguments(args)

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
