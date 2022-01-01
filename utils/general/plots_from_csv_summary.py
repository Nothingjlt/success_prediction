import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import errno
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')


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
    "mses",
    "accuracies",
    "correlations",
    "maes",
]
MODELS = [
    "model_test",
    "model_train",
    "model_worst_case"
]
COMPARISON_MODELS = [
    "null_model",
    "first_order_model",
    # "null_diff_model",
    "uniform_average",
    "linear_weighted_average",
    "square_root_weighted_average",
    "polynomial_regression",
    "uniform_periodic_average",
    "weighted_periodic_average"
]
NUMBER_OF_ITERATIONS = 30


def get_model_ids(model_names):
    model_ids = {model_name: -1 for model_name in COMPARISON_MODELS}
    for comp_model in COMPARISON_MODELS:
        for model_name in model_names:
            if not model_name.startswith(comp_model):
                continue
            contendent_model_id = int(model_name.split("_")[-1])
            if model_ids[comp_model] < contendent_model_id:
                model_ids[comp_model] = contendent_model_id
    return model_ids


def get_metric_columns(metric_type, model_ids):
    model_metric_columns = [f"{model}_{metric_type}_0" for model in MODELS]
    comparison_model_metric_columns = [
        f"{comp_model}_{metric_type}_{model_id}" for (comp_model, model_id) in model_ids.items()
    ]
    return model_metric_columns + comparison_model_metric_columns


def get_metric_keys(metric_type):
    model_keys = [
        f'{model.replace("_", " ")} {metric_type}' for model in MODELS]
    comp_model_keys = [
        f'{comp_model.replace("_", " ")} {metric_type}' for comp_model in COMPARISON_MODELS]
    return model_keys + comp_model_keys


def get_dataset_measure_iter_df(df, dataset, model_type, measure, num_of_iterations):
    dataset_measure_df = df.query(
        f'dataset=="{dataset}_{model_type}" & measure=="{measure}"'
    )
    dataset_measure_iter_df = dataset_measure_df[
        [str(i) for i in range(num_of_iterations)]
    ]
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
    return df.query(f'metric=="{metric}"')


def plot_dataset_measure(dataset_measure_iter_df, dataset, measure, metric, metric_columns, output_path):
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
    return f"{dataset} {measure} {metric}"


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
        dataset,
        "_".join([measure, metric]) + ".png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_metric_df_output_file_name(output_path, model, comparison_model, metric):
    rest_of_path = f"{model}_{metric}_vs_{comparison_model}.png"
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_model_performance_df_file_name(output_path, model, comparison_model, metric):
    rest_of_path = f"{model}_{metric.replace(' ', '_')}_performance_compared_to_{comparison_model}.png"
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_dataset_df_file_name(output_path, model, comparison_model, dataset):
    rest_of_path = os.path.join(
        "datasets",
        dataset,
        f'{model}_performance_in_{dataset}_compared_to_{comparison_model}.png'
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df(prefix, output_path):
    rest_of_path = f'Model_{prefix}_performance_in_all_datasets_compared_to_all_models.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def generate_plots(dataset_measure_metric_df, dataset, measure, metric, output_path):
    p = sns.displot(
        dataset_measure_metric_df,
        x=metric,
        hue="model",
        kde=True,
        bins=100
    )
    if metric == "maes" or metric == "mses":
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


def plot_metric_df(metric_df, model, comparison_model, metric, hue_order, output_path):
    p = sns.displot(
        metric_df,
        hue=f'{comparison_model}_comparison_score',
        x='measure',
        multiple='dodge',
        hue_order=hue_order
    )
    tag = f'{model.replace("_", " ")} {metric} performance against {comparison_model.replace("_", " ")}'
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    p.fig.axes[0].tick_params(axis='x', rotation=90)
    p.fig.canvas.start_event_loop(sys.float_info.min)
    p.savefig(
        create_metric_df_output_file_name(
            output_path,
            model,
            comparison_model,
            metric
        )
    )
    plt.close()
    return


def plot_model_performance_df(model_performance_df, model, comparison_model, metric, hue_order, output_path):
    p = sns.catplot(
        y='dataset',
        col='measure',
        hue=f'{comparison_model}_comparison_score',
        data=model_performance_df,
        kind='count',
        hue_order=hue_order
    )
    tag = f'{model.replace("_", " ")} {metric} performance compared to {comparison_model.replace("_", " ")}'
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    # p.set(title=tag)
    # p.fig.axes[0].tick_params(axis='y', rotation=45)
    p.fig.canvas.start_event_loop(sys.float_info.min)
    p.savefig(
        create_model_performance_df_file_name(
            output_path,
            model,
            comparison_model,
            metric
        )
    )
    plt.close()
    return


def plot_dataset_df(dataset_df, model, comparison_model, dataset, hue_order, output_path):
    p = sns.catplot(
        col="measure",
        row="metric",
        x=f'{comparison_model}_comparison_score',
        data=dataset_df,
        hue=f'{comparison_model}_comparison_score',
        kind='count',
        hue_order=hue_order
    )
    tag = f'{model.replace("_", " ")} performance in {dataset} compared to {comparison_model.replace("_", " ")}'
    p.fig.subplots_adjust(top=.9)
    p.fig.suptitle(tag)
    p.savefig(
        create_dataset_df_file_name(
            output_path,
            model,
            comparison_model,
            dataset
        )
    )
    plt.close()
    return


def plot_total_model_comp_df(df, hue_order, output_path):
    for model_type in ["test", "train", "worst_case"]:
        list_of_comp_models_to_concat = []
        model_comparison_names = []
        for comparison_model in COMPARISON_MODELS:
            comp_model_score = f'{comparison_model}_comparison_score'
            sliced_df = df.query(f'index.str.contains("model_{model_type}")')[
                [comp_model_score, 'measure', 'metric']
            ].rename(columns={comp_model_score: "score"})
            list_of_comp_models_to_concat.append(sliced_df)
            model_comparison_names.append(comparison_model)
        comp_df = pd.concat(
            list_of_comp_models_to_concat, keys=model_comparison_names
        ).reset_index(
        ).rename(
            columns={0: 'score', "level_0": "comparison_model"}
        )
        p = sns.catplot(
            col='measure',
            row="comparison_model",
            x="score",
            data=comp_df,
            hue='score',
            kind='count',
            hue_order=hue_order
        )
        tag = f'Model {model_type.replace("_", " ")} performance in all datasets compared to all models'
        p.fig.subplots_adjust(top=.95)
        p.fig.suptitle(tag)
        p.savefig(
            create_total_model_comp_df(
                model_type,
                output_path
            )
        )
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
    argparser.add_argument(
        "--num-of-iterations",
        type=int,
        default=NUMBER_OF_ITERATIONS,
        help="Number of iterations in model evaluation"
    )

    args = argparser.parse_args()

    df = pd.read_csv(args.file_to_parse, index_col=0)
    # df = df.rename(columns={"Unnamed: 0": "model_metric"})

    hue_order = ["won", "lost", "inconclusive"]

    plot_total_model_comp_df(df, hue_order, args.output_path)

    for metric in METRICS:
        print(f"plotting {metric} metric")
        metric_df = get_metric_df(df, metric)
        for model in MODELS:
            print(f"\tplotting {model} model")
            model_folder_path = os.path.join(
                args.output_path,
                f"{model}_comparison"
            )
            model_performance_df = df.query(f'model=="{model}"')
            metric_performance_df = model_performance_df.query(
                f'metric=="{metric}"'
            )
            for comparison_model in COMPARISON_MODELS:
                print(f"\t\tplotting {comparison_model} comparison model")
                plot_model_performance_df(
                    model_performance_df,
                    model,
                    comparison_model,
                    "all metrics",
                    hue_order,
                    os.path.join(
                        model_folder_path,
                        "comparisons"
                    )
                )

                plot_model_performance_df(
                    metric_performance_df,
                    model,
                    comparison_model,
                    metric,
                    hue_order,
                    os.path.join(
                        model_folder_path,
                        "comparisons",
                        f"compared_vs_{comparison_model}"
                    )
                )
                plot_metric_df(
                    metric_df,
                    model,
                    comparison_model,
                    metric,
                    hue_order,
                    os.path.join(model_folder_path, metric)
                )

    for dataset in DATASETS:
        print(f"plotting {dataset} dataset")
        dataset_df = model_performance_df.query(
            f'dataset.str.contains("{dataset}")'
        )
        for model in MODELS:
            print(f"\tplotting {model} model")
            model_folder_path = os.path.join(
                args.output_path,
                f"{model}_comparison"
            )
            for comparison_model in COMPARISON_MODELS:
                print(f"\t\tplotting {comparison_model} comparison model")
                plot_dataset_df(
                    dataset_df,
                    model,
                    comparison_model,
                    dataset,
                    hue_order,
                    model_folder_path
                )
    for dataset in DATASETS:
        print(f"Plotting {dataset} dataset")
        dataset_df = df.query(f'dataset=="{dataset}_{args.model_type}"')
        for measure in MEASURES:
            print(f"\tPlotting {measure} measure")
            dataset_measure_iter_df = dataset_df.query(f'measure=="{measure}"')[
                [str(i) for i in range(args.num_of_iterations)]
            ]
            for metric in METRICS:
                model_ids = get_model_ids(dataset_df.index)
                metric_columns = get_metric_columns(metric, model_ids)
                plot_dataset_measure(
                    dataset_measure_iter_df,
                    dataset,
                    measure,
                    metric,
                    metric_columns,
                    args.output_path
                )


if '__main__' == __name__:
    main()
