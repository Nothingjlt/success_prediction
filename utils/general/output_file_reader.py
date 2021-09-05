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


P_VALUE_THRESHOLD = 0.05


def split_file_name_to_measure_and_dataset(file_name):
    for m in MEASURES:
        if file_name.startswith(m):
            file_name = file_name.replace(m + "_", "")
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
        if "zero_model_tot_accuracy_" in row:
            num = int(row[len("zero_model_tot_accuracy_"):])
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


def get_worst_case_statistics(metric_name, x_name_one, x_name_two, y_name, df, alternative, should_be_greater):
    d = {}
    if should_be_greater:
        curr_df = df.loc[[x_name_one, x_name_two]].max(axis=0)
    else:
        curr_df = df.loc[[x_name_one, x_name_two]].min(axis=0)
    d[metric_name] = get_statistics(
        curr_df,
        df.loc[y_name],
        alternative=alternative,
        should_be_greater=should_be_greater
    )
    return d, curr_df


def add_values_to_df(df, line_id, col_id, value):
    df.at[line_id, col_id] = value
    return df


def prepare_file_df(root, file_name, add_train_results=False):
    acc_alternative = "two-sided"
    corr_alternative = "two-sided"
    mae_alternative = "two-sided"
    df = pd.read_csv(os.path.join(root, file_name), delimiter=',', index_col=0)
    models_index = get_latest_model_index(df)
    statistics = {}
    train_statistics = {}
    worst_case_statistics = {}
    statistics.update(
        get_metric_statistics(
            "accuracy_0",
            "model_accuracy_0",
            f"zero_model_tot_accuracy_{models_index}",
            df,
            acc_alternative,
            False
        )
    )
    statistics.update(
        get_metric_statistics(
            "correlation_0",
            "model_correlation_0",
            f"zero_model_tot_correlation_{models_index}",
            df,
            corr_alternative,
            False
        )
    )
    statistics.update(
        get_metric_statistics(
            "mae_0",
            "model_mae_0",
            f"zero_model_tot_mae_{models_index}",
            df,
            mae_alternative,
            True
        )
    )
    if add_train_results:
        train_statistics.update(
            get_metric_statistics(
                "accuracy_0",
                "model_train_accuracy_0",
                f"zero_model_tot_accuracy_{models_index}",
                df,
                acc_alternative,
                False
            )
        )
        d, worst_case_df = get_worst_case_statistics(
            "accuracy_0",
            "model_accuracy_0",
            "model_train_accuracy_0",
            f"zero_model_tot_accuracy_{models_index}",
            df,
            acc_alternative,
            False
        )
        df.loc[f"model_worst_case_accuracy_0"] = worst_case_df
        worst_case_statistics.update(d)
        train_statistics.update(
            get_metric_statistics(
                "correlation_0",
                "model_train_correlation_0",
                f"zero_model_tot_correlation_{models_index}",
                df,
                corr_alternative,
                False
            )
        )
        d, worst_case_df = get_worst_case_statistics(
            "correlation_0",
            "model_correlation_0",
            "model_train_correlation_0",
            f"zero_model_tot_correlation_{models_index}",
            df,
            corr_alternative,
            False
        )
        df.loc[f"model_worst_case_correlation_0"] = worst_case_df
        worst_case_statistics.update(d)
        train_statistics.update(
            get_metric_statistics(
                "mae_0",
                "model_train_mae_0",
                f"zero_model_tot_mae_{models_index}",
                df,
                mae_alternative,
                True
            )
        )
        d, worst_case_df = get_worst_case_statistics(
            "mae_0",
            "model_mae_0",
            "model_train_mae_0",
            f"zero_model_tot_mae_{models_index}",
            df,
            mae_alternative,
            True
        )
        df.loc[f"model_worst_case_mae_0"] = worst_case_df
        worst_case_statistics.update(d)
        df = df.loc[[item for sublist in [
            [f"model_{k}", f"model_train_{k}", f"model_worst_case_{k}", f"zero_model_tot_{k[:-1] + str(models_index)}"] for k in statistics.keys()] for item in sublist]]
    else:
        df = df.loc[[item for sublist in [
            [f"model_{k}", f"zero_model_tot_{k[:-1] + str(models_index)}"] for k in statistics.keys()] for item in sublist]]
    for k, v in statistics.items():
        zero_model_index = k.replace('_0', f"_{models_index}")
        df = add_values_to_df(
            df, f"model_{k}", "percentage_improvement", v.perc_improvement)
        df = add_values_to_df(
            df, f"model_{k}", "two_sided_paired_t_value", v.ttest_statistic)
        df = add_values_to_df(
            df, f"model_{k}", "two_sided_paired_t_test_pvalue", v.ttest_pvalue)
        df = add_values_to_df(
            df, f"model_{k}", "two_sided_wilcoxon_statistic", v.wilcoxon_statistic)
        df = add_values_to_df(
            df, f"model_{k}", "two_sided_wilcoxon_pvalue", v.wilcoxon_pvalue)

        model_won = v.perc_improvement > 0 and v.wilcoxon_pvalue < P_VALUE_THRESHOLD
        model_lost = v.perc_improvement < 0 and v.wilcoxon_pvalue < P_VALUE_THRESHOLD
        if model_won:
            model_score_against_null_model = "won"
        elif model_lost:
            model_score_against_null_model = "lost"
        else:
            model_score_against_null_model = "inconclusive"
        df = add_values_to_df(
            df, f"model_{k}", "score_against_null_model", model_score_against_null_model)

        if add_train_results:
            train_stats = train_statistics[k]
            df = add_values_to_df(
                df, f"model_train_{k}", "percentage_improvement", train_stats.perc_improvement)
            df = add_values_to_df(
                df, f"model_train_{k}", "two_sided_paired_t_value", train_stats.ttest_statistic)
            df = add_values_to_df(
                df, f"model_train_{k}", "two_sided_paired_t_test_pvalue", train_stats.ttest_pvalue)
            df = add_values_to_df(
                df, f"model_train_{k}", "two_sided_wilcoxon_statistic", train_stats.wilcoxon_statistic)
            df = add_values_to_df(
                df, f"model_train_{k}", "two_sided_wilcoxon_pvalue", train_stats.wilcoxon_pvalue)

            worst_case_stats = worst_case_statistics[k]
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "percentage_improvement", worst_case_stats.perc_improvement)
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "two_sided_paired_t_value", worst_case_stats.ttest_statistic)
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "two_sided_paired_t_test_pvalue", worst_case_stats.ttest_pvalue)
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "two_sided_wilcoxon_statistic", worst_case_stats.wilcoxon_statistic)
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "two_sided_wilcoxon_pvalue", worst_case_stats.wilcoxon_pvalue)

            model_train_won = train_stats.perc_improvement > 0 and train_stats.wilcoxon_pvalue < P_VALUE_THRESHOLD
            model_train_lost = train_stats.perc_improvement < 0 and train_stats.wilcoxon_pvalue < P_VALUE_THRESHOLD
            model_worst_case_won = worst_case_stats.perc_improvement > 0 and worst_case_stats.wilcoxon_pvalue < P_VALUE_THRESHOLD
            model_worst_case_lost = worst_case_stats.perc_improvement < 0 and worst_case_stats.wilcoxon_pvalue < P_VALUE_THRESHOLD

            if model_train_won:
                model_train_score_against_null_model = "won"
            elif model_train_lost:
                model_train_score_against_null_model = "lost"
            else:
                model_train_score_against_null_model = "inconclusive"

            if model_worst_case_won:
                model_worst_case_score_against_null_model = "won"
            elif model_worst_case_lost:
                model_worst_case_score_against_null_model = "lost"
            else:
                model_worst_case_score_against_null_model = "inconclusive"

            df = add_values_to_df(
                df, f"model_train_{k}", "score_against_null_model", model_train_score_against_null_model)
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "score_against_null_model", model_worst_case_score_against_null_model)

            potential_overfit = False
            if model_train_won and not model_won:
                potential_overfit = True

            df = add_values_to_df(
                df, f"model_{k}", "potential_overfit", potential_overfit)
            df = add_values_to_df(
                df, f"model_train_{k}", "potential_overfit", potential_overfit)
            df = add_values_to_df(
                df, f"model_worst_case_{k}", "potential_overfit", potential_overfit)

        else:
            df = add_values_to_df(
                df, f"zero_model_tot_{zero_model_index}", "percentage_improvement", v.perc_improvement)
            df = add_values_to_df(
                df, f"zero_model_tot_{zero_model_index}", "two_sided_paired_t_value", v.ttest_statistic)
            df = add_values_to_df(
                df, f"zero_model_tot_{zero_model_index}", "two_sided_paired_t_test_pvalue", v.ttest_pvalue)
            df = add_values_to_df(
                df, f"zero_model_tot_{zero_model_index}", "two_sided_wilcoxon_statistic", v.wilcoxon_statistic)
            df = add_values_to_df(
                df, f"zero_model_tot_{zero_model_index}", "two_sided_wilcoxon_pvalue", v.wilcoxon_pvalue)
    df.fillna('', inplace=True)
    measure, dataset = split_file_name_to_measure_and_dataset(file_name)
    df["measure"] = measure
    df["dataset"] = dataset
    return df


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
    argparser.add_argument('--add_train_results', type=bool,
                           help="Decide whether to print train results", nargs='?', default=False, const=True)
    args = argparser.parse_args()
    analyze_GCNRNN_files_in_outdir(
        args.folder_to_iterate, args.output_file_name, args.add_train_results)


if __name__ == '__main__':
    main()
