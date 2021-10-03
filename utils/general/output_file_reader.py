import re
import argparse
import pandas as pd
from scipy import stats
import os
from collections import namedtuple

# ACC_REGEX = "Final result: model tot accuracy: \[(?P<model_tot_acc>.*?)\], zero_model_tot_accuracy: \[(?P<zero_model_tot_acc>.*?)\], first order tot accuracy: \[(?P<first_order_model_tot_acc>.*?)\], zero model diff accuracy: \[(?P<zero_model_diff_acc>.*?)\], first order diff accuracy: \[(?P<first_order_model_diff_acc>.*?)\]"
# COR_REGEX = "Final result: model tot correlation: \[(?P<model_tot_acc>.*?)\], zero_model_tot_correlation: \[(?P<zero_model_tot_acc>.*?)\], first order tot correlation: \[(?P<first_order_model_tot_acc>.*?)\], zero model diff correlation: \[(?P<zero_model_diff_acc>.*?)\], first order diff correlation: \[(?P<first_order_model_diff_acc>.*?)\]"
# MAE_REGEX = "Final result: model tot mae: \[(?P<model_tot_acc>.*?)\], zero_model_tot_mae: \[(?P<zero_model_tot_acc>.*?)\], first order tot mae: \[(?P<first_order_model_tot_acc>.*?)\], zero model diff mae: \[(?P<zero_model_diff_acc>.*?)\], first order diff mae: \[(?P<first_order_model_diff_acc>.*?)\]"

# COL_NAMES = ['models_acc', 'null_models_acc', 'first_order_models_acc', 'null_models_diff_add', 'first_order_models_diff_acc', 'models_corr', 'null_models_corr', 'first_orders_model_corr',
#              'null_models_diff_corr', 'first_order_models_diff_corr', 'models_mae', 'null_models_mae', 'first_orders_model_mae', 'null_models_diff_mae', 'first_order_models_diff_mae']


# def read_from_file(file_name):
#     raw_string = open(file_name, 'r').read()
#     accuracies = re.findall(ACC_REGEX, raw_string)
#     correlations = re.findall(COR_REGEX, raw_string)
#     maes = re.findall(MAE_REGEX, raw_string)
#     csv_type_strings = []
#     for l in zip(accuracies, correlations, maes):
#         print('\t'.join(sum(l, ())))
#         csv_type_strings.append('\t'.join(sum(l, ())))
#     open(file_name+".csv", 'w').write("\n".join(csv_type_strings))


MEASURES = [
    "betweenness_centrality",
    "closeness_centrality",
    "general",
    "k_core",
    "load_centrality",
    "page_rank"
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
        "null_diff_model_comparison",
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
    orig_df.loc[f"{worst_case_prefix}_correlations_0", "metric"] = "correlations"
    clean_df.loc[f"{worst_case_prefix}_maes_0"] = get_worst_case_metric(
        f"{model_test_prefix}_maes_0",
        f"{model_train_prefix}_maes_0",
        clean_df,
        True
    )
    orig_df.loc[f"{worst_case_prefix}_maes_0"] = clean_df.loc[f"{worst_case_prefix}_maes_0"]
    orig_df.loc[f"{worst_case_prefix}_maes_0", "metric"] = "maes"

    orig_df.loc[orig_df.index.str.contains(f"{worst_case_prefix}"), "model"] = "model_worst_case"

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
    null_diff_model_comparison = update_model_comparison(
        clean_df,
        base_model_prefix,
        "null_diff_model",
        models_index
    )
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
        null_diff_model_comparison,
        uniform_average_model_comparison,
        linear_weighted_average_comparison,
        square_root_weighted_average_comparison,
        polynomial_regression_comparison,
        uniform_periodic_average_comparison,
        weighted_periodic_average_comparison
    )

    return all_models_comparison


def decide_if_model_won(statistics):
    if statistics.wilcoxon_pvalue > P_VALUE_THRESHOLD:
        comparison_score = "inconclusive"
    else:
        if statistics.perc_improvement > 0:
            comparison_score = "won"
        else:
            comparison_score = "lost"
    return comparison_score


def add_comparison_to_df(df, base_model_prefix, comparison_results):
    for comparison_name, comparison_result in comparison_results._asdict().items():
        for metric, statistics in comparison_result.items():
            for statistic_name, statistic_value in statistics._asdict().items():
                df = add_values_to_df(df, f"{base_model_prefix}_{metric}", f"{comparison_name}_{statistic_name}", statistic_value)
            comparison_score = decide_if_model_won(statistics)            
            df = add_values_to_df(df, f"{base_model_prefix}_{metric}", f"{comparison_name}_score", comparison_score)
    return df

def prepare_file_df(root, file_name, add_train_results=False):
    df = pd.read_csv(os.path.join(root, file_name), delimiter=',', index_col=0)
    models_index = get_latest_model_index(df)
    statistics = {}
    train_statistics = {}
    worst_case_statistics = {}
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
        output_df = add_comparison_to_df(output_df, "model_train", model_train_comparison_results)
        output_df = add_comparison_to_df(output_df_with_worst_case, "model_worst_case", model_worst_case_comparison_results)
    output_df = add_comparison_to_df(output_df, "model_test", model_test_comparison_results)
    output_df.fillna('', inplace=True)
    measure, dataset = split_file_name_to_measure_and_dataset(file_name)
    output_df["measure"] = measure
    output_df["dataset"] = dataset
    output_df["metric"] = df["metric"]
    output_df["model"] = df["model"]
    return output_df


def analyze_GCNRNN_files_in_outdir(
    out_files_dir: str = r"C:\Users\nothi\Google Drive\Studies\Courses\Thesis\from_dsi\success_prediction\out",
    output_file_name: str = r"C:\Users\nothi\Google Drive\Studies\Courses\Thesis\research proposal\all_summary_raw_data_df.txt",
    add_train_results: bool = False
):
    master_df = pd.DataFrame()
    for r, d, files in os.walk(out_files_dir):
        for f in files:
            if f.endswith("_GCNRNN.out.csv"):
                df = prepare_file_df(r, f, add_train_results)
                master_df = pd.concat([master_df, df])
    print(master_df)
    with open(output_file_name, 'w') as out_file:
        out_file.write(master_df.to_csv(sep=",").replace('\r\n', '\n'))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('output_file_name', type=str,
                           help="path to output file")
    argparser.add_argument('folder_to_iterate', type=str,
                           help="path to folder to recurse and parse")
    argparser.add_argument('--add_train_results', action='store_true',
                           help="Decide whether to print train results")
    args = argparser.parse_args()
    analyze_GCNRNN_files_in_outdir(
        args.folder_to_iterate, args.output_file_name, args.add_train_results)


if __name__ == '__main__':
    main()
