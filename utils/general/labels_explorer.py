# %%
import os
import sys
import errno
from argparse import ArgumentParser
import itertools
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')


DATASETS = {
    "ca-cit-HepPh": [
        "ca_cit_hepph"
    ],
    "cambridge_haggle": [
        "complab05",
        "infocom05",
        "infocom06_four_hours",
        "infocom06_hours",
        "infocom06",
        "intel05"
    ],
    "dnc": [
        "dnc_candidate_one",
        "dnc_candidate_two"
    ],
    "fb-wosn-friends": [
        "fb-wosn-friends_1",
        "fb-wosn-friends_2"
    ],
    "ia-digg-reply": [
        "ia-digg-reply"
    ],
    "ia-retweet-pol": [
        "ia_retweet_pol"
    ],
    "ia-slashdot-reply-dir": [
        "ia-slashdot-reply-dir"
    ],
    "reality_mining": [
        "reality_mining_daily",
        "reality_mining_monthly"
    ],
}

LABEL_TYPES = [
    "betweenness_centrality",
    "closeness_centrality",
    "k_core",
    "load_centrality",
    "page_rank",
    "general",
]


MULTI_HISTOGRAMS_COL_WRAP = 3

# %%


def load_labels(data_folder_name, data_name):
    df = pickle.load(
        open(
            "./Pickles/" + data_folder_name + "/" +
            data_name + "_labels_df" + ".pkl", "rb"
        )
    )

    return df

# %%


def plot_data_evolution(data_df, label_type, connected_vertices_filter):
    df_connected_vertices = data_df.loc[connected_vertices_filter]
    p = sns.lineplot(data=data_df,
                     x='timestep',
                     y=label_type,
                     ci=100
                     )
    p = sns.lineplot(
        data=df_connected_vertices,
        x='timestep',
        y=label_type,
        ci=100
    )
    p.legend(labels=['all vertices', 'only connected vertices'])
    return p


def process_figure(figure, tag, plot_name_to_save):
    figure.subplots_adjust(top=.9)
    figure.suptitle(tag)
    # p.set(title=tag)
    figure.canvas.start_event_loop(sys.float_info.min)
    figure.savefig(plot_name_to_save)
    plt.close()


def process_subplot(p, tag, plot_name_to_save):
    process_figure(p.figure, tag, plot_name_to_save)


def process_plot(p, tag, plot_name_to_save):
    process_figure(p.fig, tag, plot_name_to_save)


def create_average_time_evolution_plot(data_df, connected_vertices_filter, label_type, output_path, data_folder_name, data_name, additional_name=''):
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        label_type
    )
    p = plot_data_evolution(data_df, label_type, connected_vertices_filter)
    tag = f"{data_folder_name}/{data_name} {' '.join([additional_name, label_type])} evolution over timesteps"
    process_subplot(p, tag, plot_name_to_save)


def create_label_time_histogram_evolution(data_df, connected_vertices_filter, label_type, output_path, data_folder_name, data_name, additional_name=''):
    # g = sns.FacetGrid(data_df.loc[connected_vertices_filter], col='timestep', col_wrap=MULTI_HISTOGRAMS_COL_WRAP, hue='timestep')
    # g.map(sns.kdeplot, label_type)
    g = sns.displot(data_df.loc[connected_vertices_filter], x=label_type, kind='kde', hue='timestep', palette='viridis')
    tag = f"{data_folder_name}/{data_name} {' '.join([additional_name, label_type])} evolution over timesteps"
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        "_".join([label_type, 'histograms'])
    )
    process_plot(g, tag, plot_name_to_save)
    

def create_time_evolution_plots(data_df, connected_vertices_filter, output_path, data_folder_name, data_name, additional_name=''):
    for label_type in LABEL_TYPES:
        create_average_time_evolution_plot(data_df, connected_vertices_filter, label_type, output_path, data_folder_name, data_name, additional_name)
        create_label_time_histogram_evolution(data_df, connected_vertices_filter, label_type, output_path, data_folder_name, data_name, additional_name)


def create_output_file_name(base_output_path, rest_of_path):
    output_file_name = os.path.join(base_output_path, rest_of_path)
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return output_file_name


def create_data_evolution_file_name(output_path, data_folder_name, data_name, additional_name, label_type):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        "_".join([additional_name, label_type,
                  "evolution", "over", "time"]) + ".png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_centralities_correlations_name(output_path, data_folder_name, data_name, additional_name):
    rest_of_path = os.path.join(data_folder_name, data_name, f"centralities_correlations_{additional_name}.png")
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_pair_plots_name(output_path, data_folder_name, data_name, additional_name):
    rest_of_path = os.path.join(data_folder_name, data_name, f"Centralities_distributions_{additional_name.replace(' ', '_')}.png")
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def convert_df_to_float32(data_df):
    try:
        float32_data_df = data_df.reset_index().astype('float32')
    except ValueError:
        data_df['general'] = list(map(np.sum, data_df['general']))
        float32_data_df = data_df.reset_index().astype('float32')
    return float32_data_df


def get_connected_vertices_filter(data_df):
    connected_vertices_filter = data_df['general'] != 0
    return connected_vertices_filter


def get_logs_data_df(data_df, log_guard_scale=10):
    logs_data = data_df.copy()
    for label_type in LABEL_TYPES:
        min_non_zero = logs_data[label_type][~(
            logs_data[label_type] == 0)].min()
        log_guard = min_non_zero / log_guard_scale
        logs_data[label_type] = np.log(logs_data[label_type] + log_guard)
    return logs_data


def safe_kdeplot(*args, **kwargs):
    try:
        print(args, kwargs)
        p = sns.kdeplot(*args, **kwargs)
        p.figure.canvas.start_event_loop(sys.float_info.min)
    except ValueError:
        kwargs['bw_adjust'] = 30
        kwargs['color'] = 'red'
        print(args, kwargs)
        p = sns.kdeplot(*args, **kwargs)
        p.figure.canvas.start_event_loop(sys.float_info.min)


def plot_pairs(data_df, output_path, data_folder_name, data_name, additional_name):
    g = sns.PairGrid(data_df, vars=LABEL_TYPES)
    g.map_upper(sns.histplot, bins=(100, 100))
    g.map_diag(sns.histplot, bins=100, kde=True, stat="probability",
               kde_kws={'bw_adjust': 0.25, 'cut': 0})
    g.map_lower(safe_kdeplot, cut=0, fill=True)
    tag = f"Centralities distributions in {data_folder_name}/{data_name}, {additional_name}"
    plot_name_to_save = create_pair_plots_name(output_path, data_folder_name, data_name, additional_name)
    process_plot(g, tag, plot_name_to_save)


def plot_histogram(data, idx_to_node_id, tag, out_file_name):
    data = data[~(data == -np.inf)]
    data = data[~(data == np.inf)]
    if "betweenness" in tag or "load" in tag:
        p = sns.displot(data, bins=100)
    else:
        p = sns.displot(data)
    p.set(title=tag)
    p.savefig(os.path.join(out_file_name, tag + ".png"))

    kde_tag = " ".join([tag, "kde"])
    if "betweenness" in tag or "load" in tag:
        pkde = sns.displot(data, kde=True, bins=100)
    else:
        pkde = sns.displot(data, kde=True)
    pkde.set(title=kde_tag)
    pkde.savefig(os.path.join(out_file_name, kde_tag + ".png"))


def plot_correlation(data_1, data_2, idx_to_node_id, tag, out_file_name):
    if "betweenness" in tag or "load" in tag:
        p = sns.displot(x=data_1, y=data_2, rug=True,
                        cbar=True, bins=(100, 100))
    else:
        p = sns.displot(x=data_1, y=data_2, rug=True, cbar=True)
    p.set(title=tag)
    p.set_axis_labels(x_var="target", y_var="baseline")
    p.savefig(os.path.join(out_file_name, tag+".png"))


def plot_single_label(nodes_labels, idx_to_node_id, label, out_file_name):
    target_labels = nodes_labels[:, -1]
    baseline_labels = nodes_labels[:, -2]
    mask = (~np.isnan(target_labels) & ~np.isnan(baseline_labels))
    target_labels = target_labels[mask]
    baseline_labels = baseline_labels[mask]
    plot_histogram(target_labels, idx_to_node_id,
                   "_".join([label, "target"]), out_file_name)
    plot_histogram(baseline_labels, idx_to_node_id,
                   "_".join([label, "baseline"]), out_file_name)
    plot_histogram(target_labels - baseline_labels, idx_to_node_id,
                   "_".join([label, "target_minus_baseline"]), out_file_name)
    plot_correlation(target_labels, baseline_labels, idx_to_node_id, "_".join(
        [label, "baseline_target_correlation"]), out_file_name)


def generate_plots(nodes_labels, idx_to_node_id, label_types, out_file_name):
    for label_id, label in enumerate(label_types):
        plot_single_label(nodes_labels[label_id],
                          idx_to_node_id, label, out_file_name)
        # plot_single_label(np.log(nodes_labels[label_id]), idx_to_node_id, "_".join(
        #     [label, "log"]), out_file_name)


def create_centralities_correlations_plot(data_df, output_path, data_folder_name, data_name, additional_name):
    correlation_matrix = data_df[LABEL_TYPES].corr(method='spearman')
    p = sns.heatmap(correlation_matrix, annot=True, square=True, vmin=0, vmax=1)
    tag = f"Correlations between centralities in {data_folder_name}/{data_name} for {additional_name.replace('_', ' ')}"
    output_file_name = create_centralities_correlations_name(output_path, data_folder_name, data_name, additional_name)
    process_subplot(p, tag, output_file_name)

def create_centralities_correlations_plots(data_df, connected_vertices_filter, output_path, data_folder_name, data_name):
    create_centralities_correlations_plot(data_df, output_path, data_folder_name, data_name, 'all_vertices')
    create_centralities_correlations_plot(data_df.loc[connected_vertices_filter], output_path, data_folder_name, data_name, 'only_connected_vertices')


def analyze_dataset(data_folder_name, data_name, output_path, run_complex_plots=False):
    orig_data_df = load_labels(data_folder_name, data_name)
    data_df = convert_df_to_float32(orig_data_df)
    connected_vertices_filter = get_connected_vertices_filter(data_df)
    logs_data_df = get_logs_data_df(data_df)

    create_centralities_correlations_plots(data_df, connected_vertices_filter, output_path, data_folder_name, data_name)

    create_time_evolution_plots(
        logs_data_df, connected_vertices_filter, output_path, data_folder_name, data_name, 'log')
    create_time_evolution_plots(
        data_df, connected_vertices_filter, output_path, data_folder_name, data_name)

    if run_complex_plots:
        plot_pairs(data_df, output_path, data_folder_name, data_name, "only connected vertices log")
        plot_pairs(data_df[connected_vertices_filter], output_path, data_folder_name, data_name, "only connected vertices")
        plot_pairs(logs_data_df, output_path, data_folder_name, data_name, "all vertices log")
        plot_pairs(data_df, output_path, data_folder_name, data_name, "all vertices")
    return  # TODO


def load_all_datasets():
    all_data_dfs_list = []
    for folder, datasets in DATASETS.items():
        for dataset in datasets:
            new_data_df = load_labels(folder, dataset)
            new_data_df['dataset'] = f"{folder}_{dataset}"
            all_data_dfs_list.append(new_data_df)
    all_data_df = pd.concat(all_data_dfs_list)
    return all_data_dfs_list, all_data_df


def compare_all_datasets():
    all_data_dfs_list, all_data_df = load_all_datasets()


def get_args():
    argparser = ArgumentParser()
    argparser.add_argument(
        "output_path",
        type=str,
        help="path of dir to save plots to"
    )
    argparser.add_argument("--run-complex-plots", action='store_true')

    args = argparser.parse_args()
    return args


def main():
    args = get_args()
    for folder, datasets in DATASETS.items():
        for dataset in datasets:
            print(f"analyzing {folder}/{dataset}")
            analyze_dataset(
                folder,
                dataset,
                args.output_path,
                run_complex_plots=args.run_complex_plots
            )
    # compare_all_datasets()
    return


if '__main__' == __name__:
    main()

# %%
