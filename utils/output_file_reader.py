import re
import argparse
import pandas as pd
from scipy import stats
import os

from scipy.stats.stats import ttest_rel

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


def prepare_file_df(root, file_name):
    measure = file_name.split("_")[0]
    df = pd.read_csv(os.path.join(root, file_name), delimiter='\t', names=COL_NAMES).T
    statistics = {}
    statistics["acc"] = get_statistics(df.loc["models_acc"], df.loc["null_models_acc"], alternative="greater")
    statistics["corr"] = get_statistics(df.loc["models_corr"], df.loc["null_models_corr"], alternative="greater")
    statistics["mae"] = get_statistics(df.loc["models_mae"], df.loc["null_models_mae"], alternative="less")
    for k, v in statistics.items():
        df.at[f"models_{k}", "one_sided_paired_t_value"] = v[0]
        df.at[f"models_{k}", "one_sided_paired_t_test_pvalue"] = v[1]
        df.at[f"models_{k}", "wilcoxon_statistic"] = v[2]
        df.at[f"models_{k}", "wilcoxon_pvalue"] = v[3]
        df.at[f"null_models_{k}", "one_sided_paired_t_value"] = v[0]
        df.at[f"null_models_{k}", "one_sided_paired_t_test_pvalue"] = v[1]
        df.at[f"null_models_{k}", "wilcoxon_statistic"] = v[2]
        df.at[f"null_models_{k}", "wilcoxon_pvalue"] = v[3]
    df["measure"] = measure
    df["dataset"] = file_name
    return df.loc[[ item for sublist in [[f"models_{k}", f"null_models_{k}"] for k in statistics.keys()] for item in sublist]]

def analyze_file(
    out_files_dir: str = r"C:\Users\nothi\Google Drive\Studies\Courses\Thesis\from_dsi\success_prediction\out",
    output_file_name: str = r"C:\Users\nothi\Google Drive\Studies\Courses\Thesis\research proposal\all_summary_raw_data_df.txt"
):
    master_df = pd.DataFrame()
    for r, d, files in os.walk(out_files_dir):
        for f in files:
            if f.endswith(".csv"):
                df = prepare_file_df(r, f)
                master_df = pd.concat([master_df, df])
    print(master_df)
    with open(output_file_name, 'w') as out_file:
        master_df.to_csv(out_file)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('file_name', type=str, help="name of file to parse")
    args = argparser.parse_args()
    read_from_file(args.file_name)


if __name__ == '__main__':
    main()
