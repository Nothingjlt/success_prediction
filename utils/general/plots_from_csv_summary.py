import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import errno
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')


MODEL_TYPES = [
    "GCNRNN"
]
DATASETS = [
    "ca_cit_hepph",
    "complab05",
    "infocom05",
    "infocom06_four_hours",
    "infocom06_hours",
    "infocom06",
    "intel05",
    "dnc_candidate_one",
    "dnc_candidate_two",
    "fb-wosn-friends_1",
    "fb-wosn-friends_2",
    "ia-digg-reply",
    "ia_retweet_pol",
    "ia-slashdot-reply-dir",
    "reality_mining_daily",
    "reality_mining_monthly",
]
MEASURES = [
    "betweenness_centrality",
    "closeness_centrality",
    "general",
    "k_core",
    "load_centrality",
    "page_rank"
]
METRICS = [
    "accuracy",
    "correlation",
    "mae",
]
NUMBER_OF_ITERATIONS = 30


def get_zero_model_id(model_names):
    model_id = -1
    for model_name in model_names:
        if model_name.startswith("zero_model_tot_"):
            model_id = int(model_name.split("_")[-1])
            break
    return model_id


def get_metric_columns(metric_type, zero_model_id):
    return [
        f"model_{metric_type}_0",
        f"model_train_{metric_type}_0",
        f"model_worst_case_{metric_type}_0",
        f"zero_model_tot_{metric_type}_{zero_model_id}"
    ]


def get_metric_keys(metric_type):
    return [
        f"model {metric_type}",
        f"model training set {metric_type}",
        f"model worst training/test {metric_type}",
        f"zero model {metric_type}",
    ]


def get_dataset_measure_iter_df(df, dataset, model_type, measure, num_of_iterations):
    dataset_measure_df = df.query(
        f'dataset=="{dataset}_{model_type}.out.csv" & measure=="{measure}"'
    )
    dataset_measure_iter_df = dataset_measure_df[
        [str(i) for i in range(num_of_iterations)]
    ].rename(index=dataset_measure_df["metric"])
    return dataset_measure_iter_df


def get_measure_metric_df_list(measure_df, metric):
    return [
        measure_df.query(
            f'metric.str.startswith("model_{metric}_")'
        ),
        measure_df.query(
            f'metric.str.startswith("model_train_{metric}_")'
        ),
        measure_df.query(
            f'metric.str.startswith("model_worst_case_{metric}_")'
        ),
        measure_df.query(
            f'metric.str.startswith("zero_model_tot_{metric}_")'
        ),
    ]


def get_measure_df(df, measure, filter_potential_overfit=True):
    return df.query(f'measure=="{measure}" & potential_overfit{"!=" if filter_potential_overfit else "=="}True')


def get_metric_df(df, metric):
    return df.query(f'metric.str.startswith("model_{metric}_")')


def plot_dataset_measure(dataset_measure_iter_df, dataset, measure, metric, output_path):
    zero_model_id = get_zero_model_id(dataset_measure_iter_df.index)
    metric_columns = get_metric_columns(metric, zero_model_id)
    # generate_two_dim_plot(
    #     dataset_measure_iter_df.T,
    #     dataset,
    #     measure,
    #     metric,
    #     metric_columns[0],
    #     metric_columns[-1]
    # )
    list_of_metric_dfs_to_concat = [
        dataset_measure_iter_df.T[column] for column in metric_columns
    ]
    metric_keys = get_metric_keys(metric)
    dataset_measure_metric_df = pd.concat(
        list_of_metric_dfs_to_concat,
        keys=metric_keys
    ).reset_index().rename(columns={0: metric, "level_0": "model"})
    generate_plots(
        dataset_measure_metric_df,
        dataset,
        measure,
        metric,
        output_path
    )
    return


def clean_dataset_name(dataset):
    return dataset.replace('.out.csv', '')


def create_tag(dataset, measure, metric):
    return f"{clean_dataset_name(dataset)} {measure} {metric}"


def generate_two_dim_plot(dataset_measure_iter_df, dataset, measure, metric, model_metric_col_name, zero_model_metric_col_name):
    p = sns.displot(
        dataset_measure_iter_df,
        x=zero_model_metric_col_name,
        y=model_metric_col_name,
        kind="kde"
    )
    if metric == "mae":
        p.fig.axes[0].invert_xaxis()
        p.fig.axes[0].invert_yaxis()
    tag = create_tag(dataset, measure, metric)
    tag = '\n'.join([tag, 'model performance vs. zero model performance'])
    p.set(title=tag)
    return


def create_output_file_name(base_output_path, rest_of_path):
    output_file_name = os.path.join(base_output_path, rest_of_path)
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return output_file_name


def create_dataset_measure_output_file_name(output_path, dataset, measure, metric):
    rest_of_path = os.path.join(
        clean_dataset_name(dataset),
        "_".join([measure, metric]) + ".png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_metric_df_output_file_name(output_path, metric):
    rest_of_path = f"model_{metric}_vs_null_model.png"
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_model_performance_df_file_name(output_path, metric):
    rest_of_path = f"model_{metric.replace(' ', '_')}_performance_compared_to_null_model.png"
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_dataset_df_file_name(output_path, dataset):
    rest_of_path = os.path.join(
        clean_dataset_name(dataset),
        f'model_performance_in_{dataset}_compared_to_null_model.png'
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def generate_plots(dataset_measure_metric_df, dataset, measure, metric, output_path):
    try:
        p = sns.displot(
            dataset_measure_metric_df,
            x=metric,
            hue="model",
            kde=True
        )
    except:
        p = sns.displot(
            dataset_measure_metric_df,
            x=metric,
            hue="model",
            kde=True,
            bins=100
        )
    if metric == "mae":
        p.fig.axes[0].invert_xaxis()
    tag = create_tag(dataset, measure, metric)
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    # p.set(title=tag)
    p.fig.canvas.start_event_loop(sys.float_info.min)
    p.savefig(
        create_dataset_measure_output_file_name(
            output_path,
            dataset,
            measure,
            metric
        )
    )
    plt.close()
    return


def plot_metric_df(metric_df, metric, hue_order, output_path):
    p = sns.displot(
        metric_df,
        hue='score_against_null_model',
        x='measure',
        multiple='dodge',
        hue_order=hue_order
    )
    tag = f'model {metric} performance against null model'
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    p.fig.axes[0].tick_params(axis='x', rotation=90)
    p.fig.canvas.start_event_loop(sys.float_info.min)
    p.savefig(create_metric_df_output_file_name(output_path, metric))
    plt.close()
    return


def plot_model_performance_df(model_performance_df, metric, hue_order, output_path):
    p = sns.catplot(
        y="dataset",
        col='measure',
          hue='score_against_null_model',
        data=model_performance_df,
        kind='count',
        hue_order=hue_order
    )
    tag = f'model {metric} performance compared to null model'
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    # p.set(title=tag)
    # p.fig.axes[0].tick_params(axis='y', rotation=45)
    p.fig.canvas.start_event_loop(sys.float_info.min)
    p.savefig(create_model_performance_df_file_name(output_path, metric))
    plt.close()
    return


def plot_dataset_df(dataset_df, dataset, hue_order, output_path):
    p = sns.catplot(
        col="measure",
        row="metric",
        x='score_against_null_model',
        data=dataset_df,
        hue='score_against_null_model',
        kind='count',
        hue_order=hue_order
    )
    tag = f'model performance in {clean_dataset_name(dataset)} compared to null model'
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    p.savefig(create_dataset_df_file_name(output_path, dataset))
    plt.close()
    return


def main():
    argparser = ArgumentParser()
    argparser.add_argument(
        "file_to_parse",
        type=str,
        help="path to file to read and create plots for"
    )
    argparser.add_argument(
        "model_type",
        choices=MODEL_TYPES,
        help="Type of model to plot for"
    )
    argparser.add_argument(
        "output_path",
        type=str,
        help="path of dir to save plots to"
    )

    args = argparser.parse_args()

    df = pd.read_csv(args.file_to_parse)
    df = df.rename(columns={"Unnamed: 0": "metric"})

    hue_order = ["won", "lost", "inconclusive"]

    model_performance_df = df.query(
        'metric=="model_accuracy_0" | metric=="model_correlation_0" | metric=="model_mae_0"'
    )
    plot_model_performance_df(
        model_performance_df,
        "all metrics",
        hue_order,
        args.output_path
    )

    for metric in METRICS:
        metric_performance_df = model_performance_df.query(
            f'metric=="model_{metric}_0"'
        )
        plot_model_performance_df(
            metric_performance_df,
            metric,
            hue_order,
            args.output_path
        )
        metric_df = get_metric_df(df, metric)
        plot_metric_df(metric_df, metric, hue_order, args.output_path)

    for dataset in DATASETS:
        print(f"Plotting {clean_dataset_name(dataset)}")
        dataset_df = model_performance_df.query(
            f'dataset.str.contains("{dataset}")'
        )
        plot_dataset_df(dataset_df, dataset, hue_order, args.output_path)
        for measure in MEASURES:
            dataset_measure_iter_df = get_dataset_measure_iter_df(
                df,
                dataset,
                args.model_type,
                measure,
                NUMBER_OF_ITERATIONS
            )
            for metric in METRICS:
                plot_dataset_measure(
                    dataset_measure_iter_df,
                    dataset,
                    measure,
                    metric,
                    args.output_path
                )


if '__main__' == __name__:
    main()
