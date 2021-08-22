import re
import argparse
import pandas as pd
from scipy import stats
import os

ACC_REGEX = "Final result: model tot accuracy: \[(?P<model_tot_acc>.*?)\], zero_model_tot_accuracy: \[(?P<zero_model_tot_acc>.*?)\], first order tot accuracy: \[(?P<first_order_model_tot_acc>.*?)\], zero model diff accuracy: \[(?P<zero_model_diff_acc>.*?)\], first order diff accuracy: \[(?P<first_order_model_diff_acc>.*?)\]"
COR_REGEX = "Final result: model tot correlation: \[(?P<model_tot_acc>.*?)\], zero_model_tot_correlation: \[(?P<zero_model_tot_acc>.*?)\], first order tot correlation: \[(?P<first_order_model_tot_acc>.*?)\], zero model diff correlation: \[(?P<zero_model_diff_acc>.*?)\], first order diff correlation: \[(?P<first_order_model_diff_acc>.*?)\]"
MAE_REGEX = "Final result: model tot mae: \[(?P<model_tot_acc>.*?)\], zero_model_tot_mae: \[(?P<zero_model_tot_acc>.*?)\], first order tot mae: \[(?P<first_order_model_tot_acc>.*?)\], zero model diff mae: \[(?P<zero_model_diff_acc>.*?)\], first order diff mae: \[(?P<first_order_model_diff_acc>.*?)\]"

COL_NAMES = ['models_acc', 'null_models_acc', 'first_order_models_acc', 'null_models_diff_add', 'first_order_models_diff_acc', 'models_corr', 'null_models_corr', 'first_orders_model_corr',
             'null_models_diff_corr', 'first_order_models_diff_corr', 'models_mae', 'null_models_mae', 'first_orders_model_mae', 'null_models_diff_mae', 'first_order_models_diff_mae']


def read_from_file(file_name):
    raw_string = open(file_name, 'r').read()
    accuracies = re.findall(ACC_REGEX, raw_string)
    correlations = re.findall(COR_REGEX, raw_string)
    maes = re.findall(MAE_REGEX, raw_string)
    csv_type_strings = []
    for l in zip(accuracies, correlations, maes):
        print('\t'.join(sum(l, ())))
        csv_type_strings.append('\t'.join(sum(l, ())))
    open(file_name+".csv", 'w').write("\n".join(csv_type_strings))


def get_statistics(x, y, alternative):
    ttest = stats.ttest_rel(x, y, alternative=alternative)
    wil = stats.wilcoxon(x, y, alternative=alternative)
    return ttest.statistic, ttest.pvalue, wil.statistic, wil.pvalue


def get_latest_model_index(df):
    latest_index = 0
    for row in df.index.values:
        if "zero_model_tot_accuracy_" in row:
            num = int(row[len("zero_model_tot_accuracy_"):])
            if num > latest_index:
                latest_index = num
    return latest_index


def prepare_file_df(root, file_name, check_if_model_wins=True):
    if check_if_model_wins:
        acc_alternative = "greater"
        corr_alternative = "greater"
        mae_alternative = "less"
    else:
        acc_alternative = "less"
        corr_alternative = "less"
        mae_alternative = "greater"
    measure = file_name.split("_")[0]
    df = pd.read_csv(os.path.join(root, file_name), delimiter=',', index_col=0)
    models_index = get_latest_model_index(df)
    statistics = {}
    statistics["accuracy_0"] = get_statistics(
        df.loc["model_accuracy_0"],
        df.loc[f"zero_model_tot_accuracy_{models_index}"],
        alternative=acc_alternative
    )
    statistics["correlation_0"] = get_statistics(
        df.loc["model_correlation_0"],
        df.loc[f"zero_model_tot_correlation_{models_index}"],
        alternative=corr_alternative
    )
    statistics["mae_0"] = get_statistics(
        df.loc["model_mae_0"],
        df.loc[f"zero_model_tot_mae_{models_index}"],
        alternative=mae_alternative
    )
    df = df.loc[[item for sublist in [
        [f"model_{k}", f"zero_model_tot_{k[:-1] + str(models_index)}"] for k in statistics.keys()] for item in sublist]]
    for k, v in statistics.items():
        zero_model_index = k.replace('_0', f"_{models_index}")
        df.at[
            f"model_{k}",
            "one_sided_paired_t_value"
        ] = v[0]
        df.at[
            f"model_{k}",
            "one_sided_paired_t_test_pvalue"
        ] = v[1]
        df.at[
            f"model_{k}",
            "wilcoxon_statistic"
        ] = v[2]
        df.at[
            f"model_{k}",
            "wilcoxon_pvalue"
        ] = v[3]
        df.at[
            f"zero_model_tot_{zero_model_index}",
            "one_sided_paired_t_value"
        ] = v[0]
        df.at[
            f"zero_model_tot_{zero_model_index}",
            "one_sided_paired_t_test_pvalue"
        ] = v[1]
        df.at[
            f"zero_model_tot_{zero_model_index}",
            "wilcoxon_statistic"
        ] = v[2]
        df.at[
            f"zero_model_tot_{zero_model_index}",
            "wilcoxon_pvalue"
        ] = v[3]
    df["measure"] = measure
    df["dataset"] = file_name
    return df


def analyze_GCNRNN_files_in_outdir(
    out_files_dir: str = r"C:\Users\nothi\Google Drive\Studies\Courses\Thesis\from_dsi\success_prediction\out",
    output_file_name: str = r"C:\Users\nothi\Google Drive\Studies\Courses\Thesis\research proposal\all_summary_raw_data_df.txt",
    check_if_model_wins: bool = True
):
    master_df = pd.DataFrame()
    for r, d, files in os.walk(out_files_dir):
        for f in files:
            if f.endswith("_GCNRNN.out.csv"):
                df = prepare_file_df(r, f, check_if_model_wins)
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
    argparser.add_argument('check_if_model_wins', type=bool,
                           help="Decide whether to check if model wins or to check if it loses", nargs='?', default=False, const=True)
    args = argparser.parse_args()
    analyze_GCNRNN_files_in_outdir(
        args.folder_to_iterate, args.output_file_name, args.check_if_model_wins)


if __name__ == '__main__':
    main()
