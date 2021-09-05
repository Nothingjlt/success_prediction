import torch
import pickle
import numpy as np
import pandas as pd
import os
import errno
from lib.graph_measures.features_meta.features_meta import *
from utils.utils import GraphSeriesData
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import argparse

import nni


class NETSCAPETrial(metaclass=ABCMeta):
    results_type = namedtuple(
        "Results",
        [
            "model_accuracy",
            "model_correlation",
            "model_mae",
            "model_train_accuracy",
            "model_train_correlation",
            "model_train_mae",
            "zero_model_tot_accuracy",
            "zero_model_tot_correlation",
            "zero_model_tot_mae",
            "first_order_tot_accuracy",
            "first_order_tot_correlation",
            "first_order_tot_mae",
            "zero_model_diff_accuracy",
            "zero_model_diff_correlation",
            "zero_model_diff_mae",
            "first_order_diff_accuracy",
            "first_order_diff_correlation",
            "first_order_diff_mae"
        ]
    )

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

    def run_trial(self):
        print(self._params)
        learned_label = self._params["learned_label"]
        graphs, labels, graph_features = self.load_input()
        # GPU memory limited, using only subset of time steps
        graphs, labels, graph_features = self._cut_graphs_list(
            graphs, labels, graph_features)
        train, test, validation = NETSCAPETrial.train_test_split(
            graphs, train_ratio=0.7)

        model_loss_list = []
        model_accuracy_list = []
        model_correlation_list = []
        model_mae_list = []
        model_train_loss_list = []
        model_train_accuracy_list = []
        model_train_correlation_list = []
        model_train_mae_list = []
        zero_model_tot_loss_list = []
        zero_model_tot_accuracy_list = []
        zero_model_tot_mae_list = []
        first_order_tot_loss_list = []
        first_order_tot_accuracy_list = []
        first_order_tot_mae_list = []
        zero_model_diff_loss_list = []
        zero_model_diff_accuracy_list = []
        zero_model_diff_mae_list = []
        first_order_diff_loss_list = []
        first_order_diff_accuracy_list = []
        first_order_diff_mae_list = []

        graph_data = GraphSeriesData(self._params["log_guard_scale"])

        graph_data.load_data(
            graphs,
            graph_features,
            user_node_id_to_idx=self._node_id_to_idx,
            user_idx_to_node_id=self._idx_to_node_id,
            learned_label=learned_label,
            labels_list=labels,
            learn_logs=self._should_learn_logs(),
            learn_diffs=self._should_learn_diff(),
            train=train,
            test=test,
            validation=validation
        )

        model = self._get_model(graph_data)
        model.train()
        with torch.no_grad():
            (
                test_loss,
                test_accuracy,
                test_correlation,
                test_mae
            ) = model.evaluate("test", evaluate_accuracy=True, evaluate_correlation=True, evaluate_mae=True)
            (
                train_loss,
                train_accuracy,
                train_correlation,
                train_mae
            ) = model.evaluate("train", evaluate_accuracy=True, evaluate_correlation=True, evaluate_mae=True)

            (
                zero_model_tot_loss_list,
                zero_model_tot_accuracy_list,
                zero_model_tot_correlation_list,
                zero_model_tot_mae_list
            ) = graph_data.evaluate_zero_model_total_num(labels, "test")
            (
                first_order_tot_loss_list,
                first_order_tot_accuracy_list,
                first_order_tot_correlation_list,
                first_order_tot_mae_list
            ) = graph_data.evaluate_first_order_model_total_number(labels, "test")
            (
                zero_model_diff_loss_list,
                zero_model_diff_accuracy_list,
                zero_model_diff_correlation_list,
                zero_model_diff_mae_list
            ) = graph_data.evaluate_zero_model_diff(labels, "test")
            (
                first_order_diff_loss_list,
                first_order_diff_accuracy_list,
                first_order_diff_correlation_list,
                first_order_diff_mae_list
            ) = graph_data.evaluate_first_order_model_diff(labels, "test")

        if self._nni:
            nni.report_intermediate_result(
                test_accuracy - zero_model_tot_accuracy_list[-1])

        model_loss_list.append(test_loss)
        model_accuracy_list.append(test_accuracy)
        model_correlation_list.append(test_correlation)
        model_mae_list.append(test_mae)

        model_train_loss_list.append(train_loss)
        model_train_accuracy_list.append(train_accuracy)
        model_train_correlation_list.append(train_correlation)
        model_train_mae_list.append(train_mae)

        print(
            f"test loss: {np.mean([m.data.item() for m in model_loss_list]):.5f}, test accuracy: {np.mean(model_accuracy_list):.5f}, "
            f"test correlation: {np.mean(model_correlation_list):.5f}, test_mae: {np.mean(model_mae_list):.5f}"
        )
        print(
            f"train loss: {np.mean([m.data.item() for m in model_train_loss_list]):.5f}, train accuracy: {np.mean(model_train_accuracy_list):.5f}, "
            f"train correlation: {np.mean(model_train_correlation_list):.5f}, train mae: {np.mean(model_train_mae_list):.5f}"
        )
        print(
            f"zero model loss: {np.mean([m.data.item() for m in zero_model_tot_loss_list]):.5f}, zero model accuracy: {np.mean(zero_model_tot_accuracy_list):.5f}, "
            f"zero model correlation: {np.mean(zero_model_tot_correlation_list):.5f}, zero model mae: {np.mean(zero_model_tot_mae_list):.5f}"
        )
        print(
            f"first order model loss: {np.mean([m.data.item() for m in first_order_tot_loss_list]):.5f}, first order model accuracy: {np.mean(first_order_tot_accuracy_list):.5f}, "
            f"first order model correlation: {np.mean(first_order_tot_correlation_list):.5f}, first order model mae: {np.mean(first_order_tot_mae_list):.5f}"
        )
        print(
            f"zero diff model loss: {np.mean([m.data.item() for m in zero_model_diff_loss_list]):.5f}, zero diff model accuracy: {np.mean(zero_model_diff_accuracy_list):.5f}, "
            f"zero diff model correlation: {np.mean(zero_model_diff_correlation_list):.5f}, zero diff model mae: {np.mean(zero_model_diff_mae_list):.5f}"
        )
        print(
            f"first order diff model loss: {np.mean([m.data.item() for m in first_order_diff_loss_list]):.5f}, first order diff model accuracy: {np.mean(first_order_diff_accuracy_list):.5f}, "
            f"first order diff model correlation: {np.mean(first_order_diff_correlation_list):.5f}, first order diff model mae: {np.mean(first_order_diff_mae_list):.5f}"
        )
        print("\n")

        return (
            [m.data.item() for m in model_loss_list],
            model_accuracy_list,
            model_correlation_list,
            model_mae_list,
            [m.data.item() for m in model_train_loss_list],
            model_train_accuracy_list,
            model_train_correlation_list,
            model_train_mae_list,
            [m.data.item() for m in zero_model_tot_loss_list],
            zero_model_tot_accuracy_list,
            zero_model_tot_correlation_list,
            zero_model_tot_mae_list,
            [m.data.item() for m in first_order_tot_loss_list],
            first_order_tot_accuracy_list,
            first_order_tot_correlation_list,
            first_order_tot_mae_list,
            [m.data.item() for m in zero_model_diff_loss_list],
            zero_model_diff_accuracy_list,
            zero_model_diff_correlation_list,
            zero_model_diff_mae_list,
            [m.data.item() for m in first_order_diff_loss_list],
            first_order_diff_accuracy_list,
            first_order_diff_correlation_list,
            first_order_diff_mae_list
        )

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

    def run_one_test_iteration(self):

        (
            model_loss,
            model_accuracy,
            model_correlation,
            model_mae,
            model_train_loss,
            model_train_accuracy,
            model_train_correlation,
            model_train_mae,
            zero_model_tot_loss,
            zero_model_tot_accuracy,
            zero_model_tot_correlation,
            zero_model_tot_mae,
            first_order_tot_loss,
            first_order_tot_accuracy,
            first_order_tot_correlation,
            first_order_tot_mae,
            zero_model_diff_loss,
            zero_model_diff_accuracy,
            zero_model_diff_correlation,
            zero_model_diff_mae,
            first_order_diff_loss,
            first_order_diff_accuracy,
            first_order_diff_correlation,
            first_order_diff_mae
        ) = self.run_trial()

        if self._nni:
            nni.report_final_result(
                model_accuracy[0]-zero_model_tot_accuracy[-1])
        else:
            print(f"Final result (accruacy): model accuracy: {model_accuracy}, zero_model_tot_accuracy: {zero_model_tot_accuracy}, "
                  f"first order tot accuracy: {first_order_tot_accuracy}, zero model diff accuracy: {zero_model_diff_accuracy}, "
                  f"first order diff accuracy: {first_order_diff_accuracy}, model train accuracy: {model_train_accuracy}")
            print(f"Final result (correlation): model correlation: {model_correlation}, zero_model_tot_correlation: {zero_model_tot_correlation}, "
                  f"first order tot correlation: {first_order_tot_correlation}, zero model diff correlation: {zero_model_diff_correlation}, "
                  f"first order diff correlation: {first_order_diff_correlation}, model train correlation: {model_train_correlation}")
            print(f"Final result (mae): model mae: {model_mae}, zero_model_tot_mae: {zero_model_tot_mae}, "
                  f"first order tot mae: {first_order_tot_mae}, zero model diff mae: {zero_model_diff_mae}, "
                  f"first order diff mae: {first_order_diff_mae}, model train mae: {model_train_mae}")
        return (
            model_accuracy,
            model_correlation,
            model_mae,
            model_train_accuracy,
            model_train_correlation,
            model_train_mae,
            zero_model_tot_accuracy,
            zero_model_tot_correlation,
            zero_model_tot_mae,
            first_order_tot_accuracy,
            first_order_tot_correlation,
            first_order_tot_mae,
            zero_model_diff_accuracy,
            zero_model_diff_correlation,
            zero_model_diff_mae,
            first_order_diff_accuracy,
            first_order_diff_correlation,
            first_order_diff_mae
        )

    def _get_output_file_name(self):
        output_file_name = "./" + self._default_out_dir + "/" + \
            str(self._params["data_folder_name"]) + "/" + \
            "_".join([self._params["learned_label"],
                      self._params["data_name"], self._get_trial_name()]) + ".out"
        return output_file_name

    def iterate_test(self):
        results = []
        output_file_name = self._get_output_file_name()
        logger = TrialSummary(output_file_name)
        for i in range(self._params["num_iterations"]):
            print(f'Iteration number {i+1} out of {self._params["num_iterations"]}')
            results.append(self.results_type(*self.run_one_test_iteration()))
        print('-'*100)
        for result in results:
            output_string = "\n".join([f"Final result: model accuracy: {result.model_accuracy}, zero model tot accuracy: {result.zero_model_tot_accuracy}, "
                                       f"first order tot accuracy: {result.first_order_tot_accuracy}, zero model diff accuracy: {result.zero_model_diff_accuracy}, "
                                       f"first order diff accuracy: {result.first_order_diff_accuracy}, model train accuracy: {result.model_train_accuracy},",
                                       f"Final result: model correlation: {result.model_correlation}, zero model tot correlation: {result.zero_model_tot_correlation}, "
                                       f"first order tot correlation: {result.first_order_tot_correlation}, zero model diff correlation: {result.zero_model_diff_correlation}, "
                                       f"first order diff correlation: {result.first_order_diff_correlation}, model train correlation: {result.model_train_correlation}",
                                       f"Final result: model mae: {result.model_mae}, zero model tot mae: {result.zero_model_tot_mae}, "
                                       f"first order tot mae: {result.first_order_tot_mae}, zero model diff mae: {result.zero_model_diff_mae}, "
                                       f"first order diff mae: {result.first_order_diff_mae}, model train mae: {result.model_train_mae}"])
            print(output_string)
        logger.write_output(results)
        return results

    def _add_general_parser_arguments(self):
        self._argparser.add_argument("--nni", action='store_true')
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
        self._argparser.add_argument("--l1-lambda", type=float, default=0,
                                     help="L1 norm regularization weight")
        self._argparser.add_argument("--epochs", type=int, default=500,
                                     help="Number of epochs to learn")
        self._argparser.add_argument("--learning-rate", type=float, default=0.001,
                                     help="Learning rate for network")
        self._argparser.add_argument("--weight-decay", type=float, default=0,
                                     help="Weight decay regularization")
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

    def run_full_trial(self, label_to_learn=None):
        self.prepare_params()

        if label_to_learn is None or self._params['learned_label'] == self._all_labels_together:
            for learned_label in self._default_features_meta.keys():
                self.set_learned_label(learned_label)
                self.iterate_test()
        else:
            self.set_learned_label(label_to_learn)
            self.iterate_test()


class TrialSummary:
    def __init__(self, output_file_name):
        self._output_file_name = output_file_name+".csv"

    def write_output(self, results):
        df = pd.DataFrame()
        for single_test_result_index, single_test_result in enumerate(results):
            for result_type, result_list in single_test_result._asdict().items():
                for single_result_index, single_result in enumerate(result_list):
                    df.at[
                        result_type+f"_{single_result_index}",
                        single_test_result_index
                    ] = single_result
        if not os.path.exists(os.path.dirname(self._output_file_name)):
            try:
                os.makedirs(os.path.dirname(self._output_file_name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(self._output_file_name, 'w') as output_file:
            output_file.write(df.to_csv(sep=",").replace('\r\n', '\n'))
        with open(self._output_file_name, 'r') as output_file:
            print(output_file.read())
