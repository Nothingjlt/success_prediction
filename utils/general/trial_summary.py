import pandas as pd
import os
import errno
from collections import namedtuple
import pickle


class TrialSummary:
    trial_data = namedtuple(
        "trial_data",
        ["model", "train", "test", "validation"]
    )
    model_evaluation_results_type = namedtuple(
        "MODEL_RESULTS",
        [
            "losses",
            "accuracies",
            "correlations",
            "maes",
            "mses"
        ]
    )
    all_results_type = namedtuple(
        "Results",
        [
            "model_test",
            "model_train",
            "null_model",
            "first_order_model",
            "null_diff_model",
            "uniform_average",
            "linear_weighted_average",
            "square_root_weighted_average",
            "polynomial_regression",
            "uniform_periodic_average",
            "weighted_periodic_average"
        ]
    )
    def __init__(self, output_file_name):
        self._csv_output_file_name = output_file_name+".csv"
        self._bin_output_file_name_prefix = f"{output_file_name}_model"

    def write_output(self, results):
        df = pd.DataFrame()
        for single_run_result_index, single_run_result in enumerate(results):
            for model_name, model_evaluation in single_run_result._asdict().items():
                for metric_name, metric_list in model_evaluation._asdict().items():
                    for single_metric_index, single_metric in enumerate(metric_list):
                        df.at[
                            f"{model_name}_{metric_name}_{single_metric_index}",
                            single_run_result_index
                        ] = single_metric
        df["model"] = ""
        for model_name in single_run_result._asdict().keys():
            df.loc[df.index.str.contains(model_name), "model"] = model_name
        df["metric"] = ""
        for metric in single_run_result[0]._asdict().keys():
            df.loc[df.index.str.contains(metric), "metric"] = metric
        if not os.path.exists(os.path.dirname(self._csv_output_file_name)):
            try:
                os.makedirs(os.path.dirname(self._csv_output_file_name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(self._csv_output_file_name, 'w') as output_file:
            output_file.write(df.to_csv(sep=",").replace('\r\n', '\n'))
        with open(self._csv_output_file_name, 'r') as output_file:
            print(output_file.read())


    def dump_models(self, models):
        for i, model_data in enumerate(models):
            model_file_name = f"{self._bin_output_file_name_prefix}_{i}.bin"
            if not os.path.exists(os.path.dirname(model_file_name)):
                try:
                    os.makedirs(os.path.dirname(model_file_name))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(model_file_name, 'wb') as f:
                pickle.dump(model_data._asdict(), f)
