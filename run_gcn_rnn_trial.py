from utils.trials.gcn_rnn_trial import GCNRNNTrial
from lib.graph_measures.features_meta.features_meta import *


def main():
    _params = {
        "print_every_num_of_epochs": 100,
        "data_folder_name": "reality_mining",
        "data_name": "reality_mining_daily",
        "graphs_cutoff_number": 40,
        "log_guard_scale": 10,
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
        "learned_label": None,
        "number_of_iterations_per_test": 30,
    }

    trial = GCNRNNTrial(_params)

    # trial._default_features_meta = {
    #         "betweenness_centrality": FeatureMeta(
    #             BetweennessCentralityCalculator, {"betweenness"}
    #         ),
    #         "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    #         "k_core": FeatureMeta(KCoreCalculator, {"kcore"}),
    #         "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load"}),
    #         "page_rank": FeatureMeta(PageRankCalculator, {"page"}),
    #         "general": FeatureMeta(GeneralCalculator, {"gen"}),
    #     }

    trial.run_full_trial("betweenness_centrality")


if __name__ == "__main__":
    main()
