import pandas as pd
import os
import errno
from collections import namedtuple


class TrialSummary:
    model_evaluation_results_type = namedtuple(
        "MODEL_RESULTS",
        [
            "mses",
            "accuracies",
            "correlations",
            "maes"
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
        self._output_file_name = output_file_name+".csv"

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
        for model_name, model_evaluation in single_run_result._asdict().items():
            for metric_name, metric_list in model_evaluation._asdict().items():
                for single_metric_index, single_metric in enumerate(metric_list):
                    df.at[
                        f"{model_name}_{metric_name}_{single_metric_index}",
                        "metric"
                    ] = metric_name
                    df.at[
                        f"{model_name}_{metric_name}_{single_metric_index}",
                        "model"
                    ] = model_name
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
