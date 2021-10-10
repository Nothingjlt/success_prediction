from utils.trials.comparison_models_trial import ComparisonTrial
from lib.graph_measures.features_meta.features_meta import *


def main():
    _params = {}

    trial = ComparisonTrial(_params)

    trial.run_full_trial()


if __name__ == "__main__":
    main()
