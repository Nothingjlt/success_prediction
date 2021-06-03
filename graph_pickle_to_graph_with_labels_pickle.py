import pickle
from lib.graph_measures.features_meta.features_meta import *
from lib.graph_measures.features_infra.graph_features import GraphFeatures

import argparse


DEFAULT_FEATURES_META = {
    "betweenness_centrality": FeatureMeta(
        BetweennessCentralityCalculator, {"betweenness"}
    ),
    "closeness": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    "kcore": FeatureMeta(KCoreCalculator, {"kcore"}),
    "load": FeatureMeta(LoadCentralityCalculator, {"load"}),
    "pagerank": FeatureMeta(PageRankCalculator, {"page"}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
}

DEFAULT_LABEL_TO_LEARN = "kcore"

DEFAULT_OUT_DIR = "out"


def get_measures_from_graphs(
    graphs: list,
    features_meta: dict = DEFAULT_FEATURES_META,
    dir_path: str = DEFAULT_OUT_DIR,
) -> list:
    labels = []
    print("getting_measures")
    for i, g in enumerate(graphs):
        print(f"{i+1} out of {len(graphs)}")
        features = GraphFeatures(g, features_meta, dir_path)
        features.build()
        features_to_add = {}
        for k in features_meta.keys():
            features_to_add[k] = features[k].features
        labels.append(features_to_add)
    return labels


def load_input(parameters: dict):
    input_fn = "./Pickles/" + \
        str(parameters["data_folder_name"]) + \
        "/" + parameters["data_name"] + ".pkl"
    output_fn = "./Pickles/" + \
        str(parameters["data_folder_name"]) + "/" + \
        parameters["data_name"] + "_with_labels" + ".pkl"
    graphs = pickle.load(open(input_fn, "rb"))
    print(len(graphs))
    labels = get_measures_from_graphs(graphs)
    print("dumping graphs with lables")
    pickle.dump((graphs, labels), open(output_fn, "wb"))


def run_trial(parameters):
    print(parameters)
    load_input(parameters)


def prepare_params(params):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-folder-name", type=str,
                           help="Folder name for dataset to evaluate")
    argparser.add_argument("--data-name", type=str,
                           help="Data pickle file name, without the .pkl extention")

    args = argparser.parse_args()

    params.update({k: v for k, v in vars(args).items() if v is not None and (
        k == "data_folder_name" or k == "data_name")})


    return params


def main():
    _params = {
        "print_every_num_of_epochs": 100,
        "data_folder_name": "dnc",
        "data_name": "dnc_candidate_two",
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
        "number_of_iterations_per_test": 30,
    }

    _params = prepare_params(_params)
    run_trial(_params)


if __name__ == "__main__":
    main()
