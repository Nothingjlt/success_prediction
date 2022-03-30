from operator import itemgetter
import argparse
import pandas as pd
from scipy import stats
import os
import errno
from collections import namedtuple
from functools import reduce


# Model_type_name: (file_name_representation_of_model_type, should_add_comparisons_to_model)
MODEL_TYPE_CHOICES = {
    "GCNRNN": ("GCNRNN", True),
    "COMPARISON": ("comparison_data", False)
}


MEASURES = [
    "betweenness_centrality",
    "closeness_centrality",
    "general",
    "k_core",
    "load_centrality",
    "page_rank"
]


NON_RESULTS_COLUMNS_ORIG_DF = [
    'model',
    'metric'
]


STATISTICS_RESULTS = namedtuple(
    'statistics_results',
    [
        'ttest_statistic',
        'ttest_pvalue',
        'wilcoxon_statistic',
        'wilcoxon_pvalue',
        'perc_improvement'
    ]
)

ALL_MODELS_COMPARISONS = namedtuple(
    "all_models_comparison",
    [
        "null_model_comparison",
        "first_order_model_comparison",
        # "null_diff_model_comparison",
        "uniform_average_comparison",
        "linear_weighted_average_comparison",
        "square_root_weighted_average_comparison",
        "polynomial_regression_comparison",
        "uniform_periodic_average_comparison",
        "weighted_periodic_average_comparison"
    ]
)


NULL_MODEL_PREFIX = "null_model"


P_VALUE_THRESHOLD = 0.05


def split_file_name_to_measure_and_dataset(file_name):
    for m in MEASURES:
        if file_name.startswith(m):
            file_name = file_name.replace(m + "_", "").replace(".out.csv", "")
            break
    return m, file_name


def get_statistics(x, y, alternative, should_be_greater=False):
    ttest = stats.ttest_rel(x, y, alternative=alternative)
    wil = stats.wilcoxon(x, y, alternative=alternative)
    x_avg = x.mean()
    y_avg = y.mean()
    perc_improvement = 100*(x_avg - y_avg)/abs(y_avg)
    if should_be_greater:
        perc_improvement *= -1
    return STATISTICS_RESULTS(ttest.statistic, ttest.pvalue, wil.statistic, wil.pvalue, perc_improvement)


def get_latest_model_index(df):
    latest_index = 0
    for row in df.index.values:
        if f"{NULL_MODEL_PREFIX}_accuracies_" in row:
            num = int(row[len(f"{NULL_MODEL_PREFIX}_accuracies_"):])
            if num > latest_index:
                latest_index = num
    return latest_index


def get_metric_statistics(metric_name, x_name, y_name, df, alternative, should_be_greater):
    d = {}
    d[metric_name] = get_statistics(
        df.loc[x_name],
        df.loc[y_name],
        alternative=alternative,
        should_be_greater=should_be_greater
    )
    return d


def get_worst_case_model(orig_df, clean_df, model_test_prefix, model_train_prefix, worst_case_prefix):
    clean_df.loc[f"{worst_case_prefix}_mses_0"] = get_worst_case_metric(
        f"{model_test_prefix}_mses_0",
        f"{model_train_prefix}_mses_0",
        clean_df,
        True
    )
    orig_df.loc[f"{worst_case_prefix}_mses_0"] = clean_df.loc[f"{worst_case_prefix}_mses_0"]
    orig_df.loc[f"{worst_case_prefix}_mses_0", "metric"] = "mses"
    clean_df.loc[f"{worst_case_prefix}_accuracies_0"] = get_worst_case_metric(
        f"{model_test_prefix}_accuracies_0",
        f"{model_train_prefix}_accuracies_0",
        clean_df,
        False
    )
    orig_df.loc[f"{worst_case_prefix}_accuracies_0"] = clean_df.loc[f"{worst_case_prefix}_accuracies_0"]
    orig_df.loc[f"{worst_case_prefix}_accuracies_0", "metric"] = "accuracies"
    clean_df.loc[f"{worst_case_prefix}_correlations_0"] = get_worst_case_metric(
        f"{model_test_prefix}_correlations_0",
        f"{model_train_prefix}_correlations_0",
        clean_df,
        False
    )
    orig_df.loc[f"{worst_case_prefix}_correlations_0"] = clean_df.loc[f"{worst_case_prefix}_correlations_0"]
    orig_df.loc[
        f"{worst_case_prefix}_correlations_0",
        "metric"
    ] = "correlations"
    clean_df.loc[f"{worst_case_prefix}_maes_0"] = get_worst_case_metric(
        f"{model_test_prefix}_maes_0",
        f"{model_train_prefix}_maes_0",
        clean_df,
        True
    )
    orig_df.loc[f"{worst_case_prefix}_maes_0"] = clean_df.loc[f"{worst_case_prefix}_maes_0"]
    orig_df.loc[f"{worst_case_prefix}_maes_0", "metric"] = "maes"

    orig_df.loc[
        orig_df.index.str.contains(f"{worst_case_prefix}"),
        "model"
    ] = "model_worst_case"

    return orig_df, clean_df


def get_worst_case_metric(x_name_one, x_name_two, df, should_be_greater):
    if should_be_greater:
        curr_df = df.loc[[x_name_one, x_name_two]].max(axis=0)
    else:
        curr_df = df.loc[[x_name_one, x_name_two]].min(axis=0)
    return curr_df


def add_values_to_df(df, line_id, col_id, value):
    df.at[line_id, col_id] = value
    return df


def update_model_comparison(clean_df, base_model_prefix, comparison_model_prefix, models_index):
    mse_alternative = "two-sided"
    acc_alternative = "two-sided"
    corr_alternative = "two-sided"
    mae_alternative = "two-sided"
    statistics = {}
    statistics.update(
        get_metric_statistics(
            "mses_0",
            f"{base_model_prefix}_mses_0",
            f"{comparison_model_prefix}_mses_{models_index}",
            clean_df,
            mse_alternative,
            True
        )
    )
    statistics.update(
        get_metric_statistics(
            "accuracies_0",
            f"{base_model_prefix}_accuracies_0",
            f"{comparison_model_prefix}_accuracies_{models_index}",
            clean_df,
            acc_alternative,
            False
        )
    )
    statistics.update(
        get_metric_statistics(
            "correlations_0",
            f"{base_model_prefix}_correlations_0",
            f"{comparison_model_prefix}_correlations_{models_index}",
            clean_df,
            corr_alternative,
            False
        )
    )
    statistics.update(
        get_metric_statistics(
            "maes_0",
            f"{base_model_prefix}_maes_0",
            f"{comparison_model_prefix}_maes_{models_index}",
            clean_df,
            mae_alternative,
            True
        )
    )
    return statistics


def compare_results(clean_df, base_model_prefix, models_index):
    null_model_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        NULL_MODEL_PREFIX,
        models_index
    )
    first_order_model_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "first_order_model",
        0
    )
    # null_diff_model_comparison = update_model_comparison(
    #     clean_df,
    #     base_model_prefix,
    #     "null_diff_model",
    #     0
    # )
    uniform_average_model_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "uniform_average",
        0
    )
    linear_weighted_average_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "linear_weighted_average",
        0
    )
    square_root_weighted_average_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "square_root_weighted_average",
        0
    )
    polynomial_regression_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "polynomial_regression",
        0
    )
    uniform_periodic_average_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "uniform_periodic_average",
        0
    )
    weighted_periodic_average_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "weighted_periodic_average",
        0
    )

    all_models_comparison = ALL_MODELS_COMPARISONS(
        null_model_comparison,
        first_order_model_comparison,
        # null_diff_model_comparison,
        uniform_average_model_comparison,
        linear_weighted_average_comparison,
        square_root_weighted_average_comparison,
        polynomial_regression_comparison,
        uniform_periodic_average_comparison,
        weighted_periodic_average_comparison
    )

    return all_models_comparison


def decide_if_model_won(statistics, p_value=P_VALUE_THRESHOLD):
    if statistics.wilcoxon_pvalue > p_value:
        comparison_score = "inconclusive"
    else:
        if statistics.perc_improvement > 0:
            comparison_score = "won"
        elif statistics.perc_improvement < 0:
            comparison_score = "lost"
        else:
            comparison_score = "inconclusive"
    return comparison_score


def add_comparison_to_df(df, base_model_prefix, comparison_results, p_value=P_VALUE_THRESHOLD):
    for comparison_name, comparison_result in comparison_results._asdict().items():
        for metric, statistics in comparison_result.items():
            for statistic_name, statistic_value in statistics._asdict().items():
                df = add_values_to_df(
                    df,
                    f"{base_model_prefix}_{metric}",
                    f"{comparison_name}_{statistic_name}",
                    statistic_value
                )
            comparison_score = decide_if_model_won(statistics, p_value)
            df = add_values_to_df(
                df,
                f"{base_model_prefix}_{metric}",
                f"{comparison_name}_score",
                comparison_score
            )
    return df


def compare_to_comparison_models(df, add_train_results, p_value=P_VALUE_THRESHOLD):
    models_index = get_latest_model_index(df)
    output_df = df.drop(['metric', 'model'], axis=1)
    model_test_comparison_results = compare_results(
        output_df,
        "model_test",
        models_index
    )
    if add_train_results:
        model_train_comparison_results = compare_results(
            output_df,
            "model_train",
            models_index
        )
        df, output_df_with_worst_case = get_worst_case_model(
            df,
            output_df,
            "model_test",
            "model_train",
            "model_worst_case"
        )
        model_worst_case_comparison_results = compare_results(
            output_df_with_worst_case,
            "model_worst_case",
            models_index
        )
        output_df = add_comparison_to_df(
            output_df_with_worst_case,
            "model_worst_case",
            model_worst_case_comparison_results,
            p_value
        )
        output_df = add_comparison_to_df(
            output_df,
            "model_train",
            model_train_comparison_results,
            p_value
        )
    output_df = add_comparison_to_df(
        output_df,
        "model_test",
        model_test_comparison_results,
        p_value
    )
    output_df.fillna('', inplace=True)

    return output_df


def remove_null_diff_model(df):
    return df[~df.index.str.contains('null_diff_model')]


def prepare_file_df(df, measure, dataset, add_comparisons=True, add_train_results=False, p_value=P_VALUE_THRESHOLD):

    if add_comparisons:
        output_df = compare_to_comparison_models(
            df, add_train_results, p_value)
    else:
        output_df = df

    output_df = remove_null_diff_model(output_df)

    output_df["measure"] = measure
    output_df["dataset"] = dataset
    output_df["metric"] = df["metric"]
    output_df["model"] = df["model"]

    return output_df


def get_results_columns(df):
    return df.columns.drop(NON_RESULTS_COLUMNS_ORIG_DF)


def rename_columns(df, results_columns, current_column_index):
    def add_current_index_to_column_name(col_name):
        if col_name in results_columns:
            val_to_return = f"{int(col_name) + current_column_index}"
        else:
            val_to_return = col_name
        return val_to_return
    return df.rename(columns=add_current_index_to_column_name)


def merge_experiment_runs(all_experiment_results):
    all_dfs = []
    for (measure, dataset), experiment in all_experiment_results.items():
        current_column_index = 0
        current_df_list = []
        for _, df in sorted(experiment, key=itemgetter(0)):
            results_columns = get_results_columns(df)
            current_df_list.append(
                rename_columns(df, results_columns, current_column_index)
            )
            current_column_index += len(results_columns)
        current_df = reduce(
            lambda left, right: left.join(right[get_results_columns(right)]),
            current_df_list
        )
        all_dfs.append((current_df, measure, dataset))
    return all_dfs


def merge_dfs(out_files_dir, model_type):
    all_experiment_results_dict = {}
    for r, d, files in os.walk(out_files_dir):
        for f in files:
            if f.endswith(f"_{MODEL_TYPE_CHOICES[model_type][0]}.out.csv"):
                experiment_name = os.path.split(os.path.split(r)[0])[-1]
                df = pd.read_csv(
                    os.path.join(r, f), delimiter=',', index_col=0
                )
                measure, dataset = split_file_name_to_measure_and_dataset(f)
                all_experiment_results_dict.setdefault(
                    (measure, dataset),
                    []).append((experiment_name, df))
    merged_dfs = merge_experiment_runs(all_experiment_results_dict)
    return merged_dfs


def analyze_files_in_outdir(
    out_files_dir: str,
    output_file_name: str,
    model_type: str,
    add_train_results: bool = False,
    p_value: float = P_VALUE_THRESHOLD
):
    merged_dfs = merge_dfs(out_files_dir, model_type)
    all_dfs = []
    for df, measure, dataset in merged_dfs:
        analyzed_df = prepare_file_df(
            df,
            measure,
            dataset,
            add_comparisons=MODEL_TYPE_CHOICES[model_type][1],
            add_train_results=add_train_results,
            p_value=p_value
        )
        all_dfs.append(analyzed_df)
    master_df = pd.concat(all_dfs)
    print(master_df)
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file_name, 'w') as out_file:
        out_file.write(master_df.to_csv(sep=",").replace('\r\n', '\n'))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('output_file_name', type=str,
                           help="path to output file")
    argparser.add_argument('folder_to_iterate', type=str,
                           help="path to folder to recurse and parse")
    argparser.add_argument('--add-train-results', action='store_true',
                           help="Decide whether to print train results")
    argparser.add_argument('--model-type', type=str, default="GCNRNN", choices=MODEL_TYPE_CHOICES.keys(),
                           help="Type of model to read. Default to GCNRNN.")
    argparser.add_argument('--target-p-value', type=float, default=P_VALUE_THRESHOLD,
                           help="Target P-Value to determine whether model won or lost.")
    args = argparser.parse_args()
    analyze_files_in_outdir(
        args.folder_to_iterate,
        args.output_file_name,
        args.model_type,
        args.add_train_results,
        args.target_p_value
    )


if __name__ == '__main__':
    main()
