import pickle
import logging
import types
import networkx as nx
# from lib.graph_measures.features_meta.features_meta import *
from lib.graph_measures.features_meta.accelerated_features_meta import FeaturesMeta as accFeaturesMeta
from lib.graph_measures.features_meta.features_meta import FeaturesMeta as nonAccFeaturesMeta
from lib.graph_measures.features_infra.graph_features import GraphFeatures
from lib.graph_measures.loggers import PrintLogger, multi_logger

import argparse


DEFAULT_OUT_DIR = "out"


DONT_ACCELERATE = ["page_rank", "k_core", "bfs_moments"]
COMMUNICABILITY_BETWEENNESS_CENTRALITY = "communicability_betweenness_centrality"
ALL_PAIRS_SHORTEST_PATH = "all_pairs_shortest_path"
ALL_PAIRS_SHORTEST_PATH_LENGTH = "all_pairs_shortest_path_length"


def get_graph_features_from_features_meta(
        graph: nx.Graph,
        dir_path: str = DEFAULT_OUT_DIR,
        logger=None,
        features_meta=None
) -> list:
    mapping = {v: j for j, v in enumerate(graph.nodes())}
    inverse_mapping = {j: v for j, v in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph_features = GraphFeatures(
        graph,
        features_meta,
        dir_path,
        logger=logger
    )
    graph_features.build()
    features_to_add = {}
    for k in features_meta.keys():
        print(k)
        if graph_features[k].is_loaded and graph_features[k].is_relevant():
            print(f"{k} is loaded and relevant")
            if isinstance(graph_features[k].features, list):
                features_to_add[k] = {
                    inverse_mapping[i]: f for i, f in enumerate(graph_features[k].features)
                }
            elif isinstance(graph_features[k].features, dict):
                features_to_add[k] = {
                    inverse_mapping[i]: f for i, f in graph_features[k].features.items()
                }
            elif isinstance(graph_features[k].features, types.GeneratorType):
                features_to_add[k] = {
                    inverse_mapping[i]: f for i, f in dict(
                        graph_features[k].features
                    ).items()
                }
            else:
                print(
                    graph_features[k].features,
                    type(
                        graph_features[k].features
                    )
                )
                assert False
    return features_to_add


def get_graph_list_features_from_features_metas(
        graphs: list,
        dir_path: str = DEFAULT_OUT_DIR,
        logger=None,
        acc_features_meta=None,
        non_acc_features_meta=None
) -> list:
    node_level_features = []
    for i, g in enumerate(graphs):
        print(f"{i+1} out of {len(graphs)}")
        if acc_features_meta is not None:
            acc_features_to_add = get_graph_features_from_features_meta(
                g,
                dir_path=dir_path,
                logger=logger,
                features_meta=acc_features_meta
            )
        else:
            acc_features_to_add = {}
        non_acc_features_to_add = get_graph_features_from_features_meta(
            g,
            dir_path=dir_path,
            logger=logger,
            features_meta=non_acc_features_meta
        )
        node_level_features.append(
            {**acc_features_to_add, **non_acc_features_to_add}
        )
    return node_level_features


def get_measures_from_graphs(
    graphs: list,
    dir_path: str = DEFAULT_OUT_DIR,
    no_acc: bool = False,
    skip_communicability_betweenness_centrality: bool = False,
    skip_all_pairs_shortest_path: bool = False,
) -> list:
    logger = multi_logger(
        [PrintLogger("Logger", level=logging.DEBUG)],
        name=None
    )
    print("getting_measures")
    all_features_non_acc = nonAccFeaturesMeta().NODE_LEVEL
    if not no_acc:
        all_features_acc = accFeaturesMeta(gpu=True, device=0).NODE_LEVEL
        for name in DONT_ACCELERATE:
            all_features_acc.pop(name)
        all_features_non_acc_keys = list(all_features_non_acc.keys())
        for name in all_features_non_acc_keys:
            if name not in DONT_ACCELERATE:
                print(f'popping {name} from all_features_non_acc')
                all_features_non_acc.pop(name)
    else:
        all_features_acc = None
    if skip_communicability_betweenness_centrality:
        print(
            f'popping {COMMUNICABILITY_BETWEENNESS_CENTRALITY} from all_features_acc and all_features_non_acc'
        )
        if all_features_acc is not None:
            all_features_acc.pop(COMMUNICABILITY_BETWEENNESS_CENTRALITY, None)
        all_features_non_acc.pop(COMMUNICABILITY_BETWEENNESS_CENTRALITY, None)
    if skip_all_pairs_shortest_path:
        print(
            f'popping {ALL_PAIRS_SHORTEST_PATH} from all_features_acc and all_features_non_acc'
        )
        if all_features_acc is not None:
            all_features_acc.pop(ALL_PAIRS_SHORTEST_PATH, None)
        all_features_non_acc.pop(ALL_PAIRS_SHORTEST_PATH, None)
        print(
            f'popping {ALL_PAIRS_SHORTEST_PATH_LENGTH} from all_features_acc and all_features_non_acc'
        )
        if all_features_acc is not None:
            all_features_acc.pop(ALL_PAIRS_SHORTEST_PATH_LENGTH, None)
        all_features_non_acc.pop(ALL_PAIRS_SHORTEST_PATH_LENGTH, None)
    node_level_features = get_graph_list_features_from_features_metas(
        graphs,
        dir_path=dir_path,
        logger=logger,
        acc_features_meta=all_features_acc,
        non_acc_features_meta=all_features_non_acc
    )
    return node_level_features


def update_graphs(graphs):
    all_nodes_set = set()
    for g in graphs:
        all_nodes_set.update(g.nodes())

    for g in graphs:
        g.add_nodes_from(all_nodes_set)

    return graphs


def load_input(parameters: dict):
    print(parameters)
    input_fn = "./Pickles/" + \
        str(parameters["data_folder_name"]) + \
        "/" + parameters["data_name"] + ".pkl"
    output_fn = "./Pickles/" + \
        str(parameters["data_folder_name"]) + "/" + \
        parameters["data_name"] + "_with_features" + ".pkl"
    graphs = pickle.load(open(input_fn, "rb"))
    print(len(graphs))
    print("updating graphs")
    updated_graphs = update_graphs(graphs)
    node_level_features = get_measures_from_graphs(
        updated_graphs,
        no_acc=parameters['no_acc'],
        skip_communicability_betweenness_centrality=parameters[
            "skip_communicability_betweenness_centrality"],
        skip_all_pairs_shortest_path=parameters["skip_all_pairs_shortest_path"]
    )
    print("dumping updated graphs with features")
    pickle.dump((updated_graphs, node_level_features), open(output_fn, "wb"))


def prepare_params(params):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-folder-name", type=str,
                           help="Folder name for dataset to evaluate", default=params["data_folder_name"])
    argparser.add_argument("--data-name", type=str,
                           help="Data pickle file name, without the .pkl extention", default=params["data_name"])
    argparser.add_argument(
        '--no-acc', type=bool, help="Use if non-accelerated run is required", nargs='?', default=False, const=True)
    argparser.add_argument(
        '--skip-communicability-betweenness-centrality',
        type=bool,
        help="Use if need to skip computation of communicability betweenness centrality",
        nargs='?',
        default=False,
        const=True
    )
    argparser.add_argument(
        '--skip-all-pairs-shortest-path',
        type=bool,
        help="Use if need to skip computation of all pairs shortest path and all pairs shortest path length",
        nargs='?',
        default=False,
        const=True
    )

    args = argparser.parse_args()

    params.update(vars(args))

    return params


def main():
    _params = {
        "data_folder_name": "dnc",
        "data_name": "dnc_candidate_two",
        "no_acc": "False",
        "skip_communicability_betweenness_centrality": "False",
        "skip_all_pairs_shortest_path": "False"
    }

    _params = prepare_params(_params)
    load_input(_params)


if __name__ == "__main__":
    main()
