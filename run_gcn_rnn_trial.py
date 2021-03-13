import networkx as nx
import torch
import pickle
import numpy as np
from lib.graph_measures.features_meta.features_meta import *
from lib.graph_measures.features_infra.graph_features import GraphFeatures
from utils.gcn_rnn_model import Model

import argparse

import ast
import nni

DEFAULT_FEATURES_META = {
    # "betweenness_centrality": FeatureMeta(
    #     BetweennessCentralityCalculator, {"betweenness"}
    # ),
    # "kcore": FeatureMeta(KCoreCalculator, {"kcore"}),
    # "load": FeatureMeta(LoadCentralityCalculator, {"load"}),
    # "pagerank": FeatureMeta(PageRankCalculator, {"page"}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
}

DEFAULT_LABEL_TO_LEARN = "general"

DEFAULT_OUT_DIR = "out"


NNI = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


def get_measures_from_graphs(
    graphs: list,
    features_meta: dict = DEFAULT_FEATURES_META,
    dir_path: str = DEFAULT_OUT_DIR,
) -> list:
    labels = []
    for g in graphs:
        features = GraphFeatures(g, features_meta, dir_path)
        features.build()
        labels.append(features)
    return labels


def load_input(parameters: dict):
    graphs = pickle.load(
        open(
            "./pickles/" + str(parameters["data_name"]) +
            "/ia_retweet_pol.pkl", "rb"
        )
    )
    # labels = pickle.load(open("./pickles/" + str(parameters["data_name"]) + "/dnc_with_labels_candidate_one.pkl", "rb"))
    labels = get_measures_from_graphs(graphs)
    all_nodes = set()
    for g in graphs:
        all_nodes.update(g.nodes())
    feature_mx = torch.eye(len(all_nodes))
    adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]
    return graphs, labels, feature_mx, adjacency_matrices


def run_trial(parameters):
    global NNI
    print(parameters)
    learned_label = parameters["learned_label"]
    graphs, labels, feature_mx, adjacency_matrices = load_input(parameters)
    graphs_cutoff_number = parameters["graphs_cutoff_number"] # GPU memory limited, using only 4 time steps
    graphs = graphs[-graphs_cutoff_number:] 
    labels = labels[-graphs_cutoff_number:]
    train, test, validation = train_test_split(graphs, train_ratio=0.7)


    model_loss_list = []
    model_accuracy_list = []
    model_tot_accuracy_list = []
    zero_model_tot_loss_list = []
    zero_model_tot_accuracy_list = []
    first_order_tot_loss_list = []
    first_order_tot_accuracy_list = []
    zero_model_diff_loss_list = []
    zero_model_diff_accuracy_list = []
    first_order_diff_loss_list = []
    first_order_diff_accuracy_list = []

    model = Model(NNI, parameters)
    model.load_data(
        graphs,
        feature_mx,
        labels[-2],
        labels[-1],
        learned_label,
        train,
        test,
        validation,
    )
    model.train()
    with torch.no_grad():
        test_loss, test_accuracy, test_tot_accuracy = model.evaluate(
            "testtrain", evaluate_accuracy=True)

        zero_model_tot_loss_list, zero_model_tot_accuracy_list = model.evaluate_zero_model_total_num(
            labels, "testtrain")
        first_order_tot_loss_list, first_order_tot_accuracy_list = model.evaluate_first_order_model_total_number(
            labels, "testtrain")
        zero_model_diff_loss_list, zero_model_diff_accuracy_list = model.evaluate_zero_model_diff(
            labels, "testtrain")
        first_order_diff_loss_list, first_order_diff_accuracy_list = model.evaluate_first_order_model_diff(
            labels, "testtrain")

    if NNI:
        nni.report_intermediate_result(
            test_tot_accuracy - zero_model_tot_accuracy_list[-1])

    model_loss_list.append(test_loss)
    model_accuracy_list.append(test_accuracy)
    model_tot_accuracy_list.append(test_tot_accuracy)

    print(
        f"test loss: {np.mean([m.data.item() for m in model_loss_list]):.5f}, test_tot_accuracy: {np.mean(model_tot_accuracy_list):.5f}, test_diff_accuracy: {np.mean(model_accuracy_list):.5f}")
    print(
        f"zero model loss: {np.mean([m.data.item() for m in zero_model_tot_loss_list]):.5f}, zero model accuracy: {np.mean(zero_model_tot_accuracy_list):.5f}")
    print(
        f"first order model loss: {np.mean([m.data.item() for m in first_order_tot_loss_list]):.5f}, first order model accuracy: {np.mean(first_order_tot_accuracy_list):.5f}")
    print(
        f"zero diff model loss: {np.mean([m.data.item() for m in zero_model_diff_loss_list]):.5f}, zero diff model accuracy: {np.mean(zero_model_diff_accuracy_list):.5f}")
    print(
        f"first order diff model loss: {np.mean([m.data.item() for m in first_order_diff_loss_list]):.5f}, first order diff model accuracy: {np.mean(first_order_diff_accuracy_list):.5f}")
    print("\n")

    return (
        [m.data.item() for m in model_loss_list],
        model_accuracy_list,
        model_tot_accuracy_list,
        [m.data.item() for m in zero_model_tot_loss_list],
        zero_model_tot_accuracy_list,
        [m.data.item() for m in first_order_tot_loss_list],
        first_order_tot_accuracy_list,
        [m.data.item() for m in zero_model_diff_loss_list],
        zero_model_diff_accuracy_list,
        [m.data.item() for m in first_order_diff_loss_list],
        first_order_diff_accuracy_list
    )


def main():
    global NNI
    _params = {
        "print_every_num_of_epochs": 100,
        "data_name": "ia-retweet-pol",
        "graphs_cutoff_number": 2,
        "l1_lambda": 0,
        "epochs": 500,
        "gcn_dropout_rate": 0.7,
        "lstm_dropout_rate": 0,
        "gcn_hidden_sizes": [10, 10, 10, 10, 10, 10, 10, 10, 10],
        "learning_rate": 0.001,
        "weight_decay": 0,
        "gcn_latent_dim": 5,
        "lstm_hidden_size": 10,
        "lstm_num_layers": 1,
        "learned_label": DEFAULT_LABEL_TO_LEARN,
    }
    l1_lambda = [0, 1e-7]
    epochs = [500]
    gcn_dropout_rate = [0.3, 0.5]
    gcn_hidden_sizes = [[100, 100], [200, 200]]
    learning_rate = [1e-3, 1e-2, 3e-2]
    weight_decay = [5e-2, 1e-2]
    gcn_latent_dim = [50, 100]
    lstm_hidden_size = [50, 100]
    results = []

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--nni", action='store_true')

    args = argparser.parse_args()

    NNI = args.nni

    if NNI:
        p = nni.get_next_parameter()
        p["gcn_hidden_sizes"] = ast.literal_eval(p["gcn_hidden_sizes"])
        _params.update(p)

    (
        model_loss,
        model_accuracy,
        model_tot_accuracy,
        zero_model_tot_loss,
        zero_model_tot_accuracy,
        first_order_tot_loss,
        first_order_tot_accuracy,
        zero_model_diff_loss,
        zero_model_diff_accuracy,
        first_order_diff_loss,
        first_order_diff_accuracy
    ) = run_trial(_params)

    if NNI:
        nni.report_final_result(
            model_tot_accuracy[0]-zero_model_tot_accuracy[-1])
    else:
        print(f"Final result: model tot accuracy: {model_tot_accuracy}, zero_model_tot_accuracy: {zero_model_tot_accuracy}, "
              f"first order tot accuracy: {first_order_tot_accuracy}, zero model diff accuracy: {zero_model_diff_accuracy}, "
              f"first order diff accuracy: {first_order_diff_accuracy}")


if __name__ == "__main__":
    main()
