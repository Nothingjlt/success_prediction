import networkx as nx
import torch
import pickle
import numpy as np
import pandas as pd
from lib.graph_measures.features_meta.features_meta import *
from lib.graph_measures.features_infra.graph_features import GraphFeatures
from utils.utils import GraphSeriesData
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import argparse

import ast
import nni


class NETSCAPETrial(metaclass=ABCMeta):
    results_type = namedtuple(
        "Results",
        [
            "model_accuracy",
            "model_correlation",
            "model_mae",
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

    def __init__(self, parameters, seed=None):
        seed_to_set = np.random.randint(0, 2 ^ 32) if seed is None else seed
        self._set_seed(seed_to_set)
        self._params = parameters
        self._default_features_meta = {
            "betweenness_centrality": FeatureMeta(
                BetweennessCentralityCalculator, {"betweenness"}
            ),
            "closeness": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
            "kcore": FeatureMeta(KCoreCalculator, {"kcore"}),
            "load": FeatureMeta(LoadCentralityCalculator, {"load"}),
            "pagerank": FeatureMeta(PageRankCalculator, {"page"}),
            "general": FeatureMeta(GeneralCalculator, {"gen"}),
        }

        self._default_label_to_learn = "kcore"
        self._default_out_dir = "out"
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
        os.environ['PYTHONHASHSEED']=str(self._seed)


    @staticmethod
    def train_test_split(graphs, train_ratio):
        prev_graph = graphs[0]
        all_intersection = np.array(prev_graph.nodes())
        for g in graphs[1:]:
            all_intersection = np.intersect1d(all_intersection, g.nodes())
            prev_graph = g

        print("# of nodes that appear at all timestamps", len(all_intersection))
        val_test_inds = np.random.choice(
            all_intersection, round(len(all_intersection) * (1-train_ratio)), replace=False
        )
        test, validation = np.array_split(val_test_inds, 2)
        train = np.setdiff1d(np.array(all_intersection), val_test_inds)
        e = 0
        return train, test, validation

    def load_input(self):
        graphs, labels = pickle.load(
            open(
                "./Pickles/" + str(self._params["data_folder_name"]) +
                "/" + self._params["data_name"] + "_with_labels" + ".pkl", "rb"
            )
        )
        all_nodes = set()
        for g in graphs:
            all_nodes.update(g.nodes())

        self._all_nodes_list = sorted(all_nodes)

        self._node_id_to_idx = {x: i for i,
                                x in enumerate(self._all_nodes_list)}
        self._idx_to_node_id = {i: x for i,
                                x in enumerate(self._all_nodes_list)}

        feature_mx = self._get_feature_matrix(graphs)

        return graphs, labels, feature_mx

    def run_trial(self):
        print(self._params)
        learned_label = self._params["learned_label"]
        graphs, labels, feature_mx = self.load_input()
        # GPU memory limited, using only subset of time steps
        graphs, labels = self._cut_graphs_list(graphs, labels)
        train, test, validation = NETSCAPETrial.train_test_split(
            graphs, train_ratio=0.7)

        model_loss_list = []
        model_accuracy_list = []
        model_correlation_list = []
        model_mae_list = []
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
            feature_mx,
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

        print(
            f"test loss: {np.mean([m.data.item() for m in model_loss_list]):.5f}, test_accuracy: {np.mean(model_accuracy_list):.5f}, test_correlation: {np.mean(model_correlation_list):.5f}, test_mae: {np.mean(model_mae_list):.5f}")
        print(
            f"zero model loss: {np.mean([m.data.item() for m in zero_model_tot_loss_list]):.5f}, zero model accuracy: {np.mean(zero_model_tot_accuracy_list):.5f}, zero model correlation: {np.mean(zero_model_tot_correlation_list):.5f}, zero model mae: {np.mean(zero_model_tot_mae_list):.5f}")
        print(
            f"first order model loss: {np.mean([m.data.item() for m in first_order_tot_loss_list]):.5f}, first order model accuracy: {np.mean(first_order_tot_accuracy_list):.5f}, first order model correlation: {np.mean(first_order_tot_correlation_list):.5f}, first order model mae: {np.mean(first_order_tot_mae_list):.5f}")
        print(
            f"zero diff model loss: {np.mean([m.data.item() for m in zero_model_diff_loss_list]):.5f}, zero diff model accuracy: {np.mean(zero_model_diff_accuracy_list):.5f}, zero diff model correlation: {np.mean(zero_model_diff_correlation_list):.5f}, zero diff model mae: {np.mean(zero_model_diff_mae_list):.5f}")
        print(
            f"first order diff model loss: {np.mean([m.data.item() for m in first_order_diff_loss_list]):.5f}, first order diff model accuracy: {np.mean(first_order_diff_accuracy_list):.5f}, first order diff model correlation: {np.mean(first_order_diff_correlation_list):.5f}, first order diff model mae: {np.mean(first_order_diff_mae_list):.5f}")
        print("\n")

        return (
            [m.data.item() for m in model_loss_list],
            model_accuracy_list,
            model_correlation_list,
            model_mae_list,
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

    def _get_feature_matrix(self, graphs):
        # Must be called after self._node_id_to_idx and self._idx_to_node_id are set!
        # features_meta = {
        #     "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),  # Directed
        #     "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),  # Any
        #     "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),  # Any
        #     "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
        #                                                           {"communicability"}),  # Undirected
        #     "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),  # Any
        #     "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),  # Undirected (due to a code limitation)
        #     "flow": FeatureMeta(FlowCalculator, {}),  # Directed
        #     "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),  # Directed (but works for any)
        #     "louvain": FeatureMeta(LouvainCalculator, {"lov"}),  # Undirected
        #     "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),  # Any
        #     "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),  # Any
        # }
        # all_features_list = []
        # for g in graphs:
        #     new_g = nx.Graph.copy(g)
        #     new_g.add_nodes_from(self._all_nodes_list)
        #     features = GraphFeatures(new_g, features_meta, self._default_out_dir)
        #     features.build()
        #     features_to_add = {}
        #     for k in features_meta.keys():
        #         if features[k].is_relevant():
        #             features_to_add[k] = features[k].features
        #     all_features_list.append(features_to_add)
        return torch.eye(len(self._all_nodes_list))

    def run_one_test_iteration(self, params):

        l1_lambda = [0, 1e-7]
        epochs = [500]
        gcn_dropout_rate = [0.3, 0.5]
        gcn_hidden_sizes = [[100, 100], [200, 200]]
        learning_rate = [1e-3, 1e-2, 3e-2]
        weight_decay = [5e-2, 1e-2]
        gcn_latent_dim = [50, 100]
        lstm_hidden_size = [50, 100]
        results = []

        (
            model_loss,
            model_accuracy,
            model_correlation,
            model_mae,
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
                  f"first order diff accuracy: {first_order_diff_accuracy}")
            print(f"Final result (correlation): model correlation: {model_correlation}, zero_model_tot_correlation: {zero_model_tot_correlation}, "
                  f"first order tot correlation: {first_order_tot_correlation}, zero model diff correlation: {zero_model_diff_correlation}, "
                  f"first order diff correlation: {first_order_diff_correlation}")
            print(f"Final result (mae): model mae: {model_mae}, zero_model_tot_mae: {zero_model_tot_mae}, "
                  f"first order tot mae: {first_order_tot_mae}, zero model diff mae: {zero_model_diff_mae}, "
                  f"first order diff mae: {first_order_diff_mae}")
        return (
            model_accuracy,
            model_correlation,
            model_mae,
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

    def iterate_test(self):
        results = []
        output_file_name = "./" + self._default_out_dir + "/" + \
            str(self._params["data_folder_name"]) + "/" + \
            "_".join([self._params["learned_label"],
                      self._params["data_name"], self._get_trial_name()]) + ".out"
        logger = TrialSummary(output_file_name)
        for i in range(self._params["number_of_iterations_per_test"]):
            results.append(self.results_type(
                *self.run_one_test_iteration(self._params)))
        print('-'*100)
        for result in results:
            output_string = "\n".join([f"Final result: model accuracy: {result.model_accuracy}, zero_model_tot_accuracy: {result.zero_model_tot_accuracy}, "
                                       f"first order tot accuracy: {result.first_order_tot_accuracy}, zero model diff accuracy: {result.zero_model_diff_accuracy}, "
                                       f"first order diff accuracy: {result.first_order_diff_accuracy}",
                                       f"Final result: model correlation: {result.model_correlation}, zero_model_tot_correlation: {result.zero_model_tot_correlation}, "
                                       f"first order tot correlation: {result.first_order_tot_correlation}, zero model diff correlation: {result.zero_model_diff_correlation}, "
                                       f"first order diff correlation: {result.first_order_diff_correlation}",
                                       f"Final result: model mae: {result.model_mae}, zero_model_tot_mae: {result.zero_model_tot_mae}, "
                                       f"first order tot mae: {result.first_order_tot_mae}, zero model diff mae: {result.zero_model_diff_mae}, "
                                       f"first order diff mae: {result.first_order_diff_mae}"])
            print(output_string)
        logger.write_output(results)
        return results

    def _add_general_parser_arguments(self):
        self._argparser.add_argument("--nni", action='store_true')
        self._argparser.add_argument("--seed", type=int,
                                     help="Optional random seed for run")
        self._argparser.add_argument("--data-folder-name", type=str,
                                     help="Folder name for dataset to evaluate")
        self._argparser.add_argument("--data-name", type=str,
                                     help="Data pickle file name, without the .pkl extention")

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

        self._params.update({k: v for k, v in vars(args).items() if v is not None and (
            k == "data_folder_name" or k == "data_name")})
        
        if args.seed is not None:
            self._set_seed(args.seed)

        self._nni = args.nni

        self._update_specific_parser_arguments(args)

        return

    def set_learned_label(self, learned_label):
        self._params["learned_label"] = learned_label

    def run_full_trial(self):
        self.prepare_params()

        for learned_label in self._default_features_meta.keys():
            self.set_learned_label(learned_label)
            self.iterate_test()


class TrialSummary:
    def __init__(self, output_file_name):
        self._output_file_name = output_file_name+".csv"

    def write_output(self, results):
        df = pd.DataFrame()
        for single_test_result_index, single_test_result in enumerate(results):
            for result_type, result_list in single_test_result._asdict().items():
                for single_result_index, single_result in enumerate(result_list):
                    df.at[result_type+f"_{single_result_index}",
                          single_test_result_index] = single_result
        with open(self._output_file_name, 'w') as output_file:
            df.to_csv(output_file, sep=",")
