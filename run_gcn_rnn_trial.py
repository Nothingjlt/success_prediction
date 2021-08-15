from utils.trials.gcn_rnn_trial import GCNRNNTrial
from lib.graph_measures.features_meta.features_meta import *


def main():
    _params = {}

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
