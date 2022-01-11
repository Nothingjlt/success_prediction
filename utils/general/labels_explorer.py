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


DATASETS = {
    "reality_mining": [
        "reality_mining_daily",
        "reality_mining_monthly"
    ],
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
}

LABEL_TYPES = [
    "betweenness_centrality",
    "load_centrality",
    "k_core",
    "general",
    "closeness_centrality",
    "page_rank",
]


MULTI_HISTOGRAMS_COL_WRAP = 3

# %%


# %%

def plotting_definitions(matplotlib_backend):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    matplotlib.use(matplotlib_backend)


def load_labels(data_folder_name, data_name):
    with open(
        f"./Pickles/{data_folder_name}/{data_name}_labels_df.pkl",
        "rb"
    ) as f:
        df = pickle.load(f)
    return df

# %%


def plot_data_evolution(data_df, label_type, connected_vertices_filter):
    df_connected_vertices = data_df.loc[connected_vertices_filter]
    p = sns.lineplot(
        data=data_df,
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


def process_figure(
    figure,
    axes,
    tag,
    plot_name_to_save,
    x_label='',
    y_label='',
    z_label=''
):
    if x_label:
        axes.set_xlabel(x_label)
    if y_label:
        axes.set_ylabel(y_label)
    if z_label:
        axes.set_zlabel(z_label)
    with open(f"{plot_name_to_save[:-4]}_title.txt", 'w') as f:
        f.write(tag)
    figure.canvas.start_event_loop(sys.float_info.min)
    figure.savefig(plot_name_to_save, bbox_inches="tight")
    plt.close()


def process_subplot(p, tag, plot_name_to_save, x_label='', y_label='', z_label=''):
    process_figure(p.figure, p.axes, tag, plot_name_to_save,
                   x_label=x_label, y_label=y_label, z_label=z_label)


def process_plot(p, tag, plot_name_to_save, x_label='', y_label='', z_label=''):
    process_figure(p.fig, p.ax, tag, plot_name_to_save,
                   x_label=x_label, y_label=y_label, z_label=z_label)


def create_average_time_evolution_plot(
    data_df,
    connected_vertices_filter,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        label_type
    )
    p = plot_data_evolution(data_df, label_type, connected_vertices_filter)
    tag = f"{data_folder_name}/{data_name}" \
        f" {' '.join([additional_name, label_type])} evolution over timesteps"
    process_subplot(p, tag, plot_name_to_save)


def get_log_guard(vector_to_guard, log_guard_scale=10):
    min_non_zero = vector_to_guard[~(vector_to_guard == 0)].min()
    log_guard = min_non_zero / log_guard_scale
    return log_guard


def plot_surface_histograms(
    timesteps,
    all_bins,
    histograms,
    occurance_log_guard,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    time_steps, histogram_bins = np.meshgrid(
        timesteps.astype('int'), all_bins[:-1])
    occurance = np.log(np.array(histograms).T + occurance_log_guard)
    surf = ax.plot_surface(
        histogram_bins,
        time_steps,
        occurance,
        linewidth=0,
        antialiased=True,
        cmap=sns.cm.rocket
    )

    tag = f"{data_folder_name}/{data_name} " \
        f"{' '.join([additional_name, label_type])} 3D evolution over timesteps"
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        "_".join([label_type, '3D', 'histograms'])
    )
    is_diff = 'diff' in additional_name
    process_subplot(
        surf,
        tag,
        plot_name_to_save,
        f"{'sym' if is_diff else ''}log("\
            f"{'diff ' if is_diff else ''}{label_type.replace('_', ' ')})",
        "timestep",
        "log(occurance)"
    )

    return


def get_histograms_for_all_timesteps(
    data_df,
    connected_vertices_filter,
    label_type,
    calc_symlog=False
):
    timesteps = data_df['timestep'].unique()
    df_connected_vertices = data_df.loc[connected_vertices_filter]
    hist_data_df = df_connected_vertices.copy()
    if calc_symlog:
        label_center_log, hist_symlog = symlog(hist_data_df[label_type])
        hist_data_df[label_type] = hist_symlog
    else:
        log_guard = get_log_guard(
            df_connected_vertices[label_type],
            log_guard_scale=10
        )
        hist_data_df[label_type] = np.log(hist_data_df[label_type] + log_guard)
    all_hist, all_bins = np.histogram(
        hist_data_df[label_type],
        bins='auto'
    )
    histograms = []
    for t in timesteps:
        current_timestep_labels = hist_data_df.loc[
            df_connected_vertices['timestep'] == t
        ][label_type]
        new_histogram, _ = np.histogram(
            current_timestep_labels,
            bins=all_bins
        )
        histograms.append(new_histogram)
    return timesteps, all_hist, all_bins, histograms


def create_surface_time_histograms(
    data_df,
    connected_vertices_filter,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    calc_symlog = 'diff' in additional_name
    (
        timesteps,
        all_hist,
        all_bins,
        histograms
    ) = get_histograms_for_all_timesteps(
        data_df,
        connected_vertices_filter,
        label_type,
        calc_symlog=calc_symlog
    )

    if len(timesteps) < 2:
        return

    occurance_log_guard = np.min(
        list(
            map(
                lambda x: get_log_guard(x, log_guard_scale=10),
                histograms
            )
        )
    )

    plot_surface_histograms(
        timesteps,
        all_bins,
        histograms,
        occurance_log_guard,
        label_type,
        output_path,
        data_folder_name,
        data_name,
        additional_name=additional_name
    )

    return


def symlog_shift(arr, shift=0):
    # shift array-like to symlog array with shift
    logv = np.abs(arr)*(np.exp(shift))
    logv.loc[logv < 1.] = 1.
    logv = np.sign(arr)*np.log(logv)
    return logv


def symlog(arr):
    min_abs_label = get_log_guard(np.abs(arr), log_guard_scale=2)
    label_center_log = -np.log(min_abs_label)
    symlog = symlog_shift(arr, shift=label_center_log)
    symlog.loc[arr > 0] = symlog.loc[arr > 0] - label_center_log
    symlog.loc[arr < 0] = symlog.loc[arr < 0] + label_center_log
    return label_center_log, symlog


def symlog_shift_ticks(tks1, tks2, tks3, shift=0):
    # generate the tick position and the corresponding tick labels in symlog scale with shift
    # tks1, tks2, tks3: tick values in log scale

    # tick positions to show in graph
    tkps = [v-shift for v in tks1]+tks2+[v+shift for v in tks3]
    # tkck labels in str
    tkls = ['$-10^{%d}$' % (v) for v in tks1] + \
        [''] + \
        ['$10^{%d}$' % (v) for v in tks3]
    return tkps, tkls


def create_label_time_histogram_evolution(
    data_df,
    connected_vertices_filter,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    # g = sns.FacetGrid(data_df.loc[connected_vertices_filter], col='timestep', col_wrap=MULTI_HISTOGRAMS_COL_WRAP, hue='timestep')
    # g.map(sns.kdeplot, label_type)
    log_scale = 'log' in additional_name
    diffs = 'diff' in additional_name
    if log_scale:
        if diffs:
            label_center_log, arr = symlog(data_df[label_type])
        else:
            log_guard = get_log_guard(data_df[label_type], log_guard_scale=10)
            arr = data_df[label_type] + log_guard
    else:
        arr = data_df[label_type]

    data_df_to_plot = data_df.copy()
    data_df_to_plot[label_type] = arr
    try:
        g = sns.displot(
            data_df_to_plot.loc[connected_vertices_filter],
            x=label_type,
            kind='kde',
            hue='timestep',
            palette='viridis',
            log_scale=(log_scale and not diffs, log_scale and not diffs)
        )
    except np.linalg.LinAlgError:
        return
    if log_scale and diffs:
        plt.yscale('log')
        g.set_xlabels(f'symlog(diff {label_type})')
        plt.text(0, 10**(-6), '$-e^{%.2f}$~$e^{%.2f}$' %
                 (-label_center_log, -label_center_log), va='bottom', ha='center')  # TODO fix transform
        plt.axvline(x=0, ymin=-1.2, ymax=1.2,
                    ls='--', lw=0.5, color='k')
    tag = f"{data_folder_name}/{data_name}" \
        f" {' '.join([additional_name, label_type])} evolution over timesteps"
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        "_".join([label_type, 'histograms'])
    )
    process_plot(g, tag, plot_name_to_save)


def create_time_evolution_plots(
    data_df,
    connected_vertices_filter,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''


):
    for label_type in LABEL_TYPES:
        if not 'log' in additional_name:
            create_surface_time_histograms(
                data_df,
                connected_vertices_filter,
                label_type,
                output_path,
                data_folder_name,
                data_name,
                additional_name
            )
            create_average_time_evolution_plot(
                data_df,
                connected_vertices_filter,
                label_type,
                output_path,
                data_folder_name,
                data_name,
                additional_name
            )
        create_label_time_histogram_evolution(
            data_df,
            connected_vertices_filter,
            label_type,
            output_path,
            data_folder_name,
            data_name,
            additional_name
        )


def create_output_file_name(base_output_path, rest_of_path):
    output_file_name = os.path.join(base_output_path, rest_of_path)
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return output_file_name


def create_data_evolution_file_name(
    output_path,
    data_folder_name,
    data_name,
    additional_name,
    label_type
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"{additional_name}_{label_type}_evolution_over_time.png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_centralities_correlations_name(
    output_path,
    data_folder_name,
    data_name,
    additional_name
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"centralities_correlations_{additional_name}.png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_pair_plots_name(
    output_path,
    data_folder_name,
    data_name,
    additional_name
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"Centralities_distributions_{additional_name.replace(' ', '_')}.png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def convert_df_to_float64(data_df):
    try:
        float64_data_df = data_df.reset_index().astype('float64')
    except ValueError:
        data_df['general'] = list(map(np.sum, data_df['general']))
        float64_data_df = data_df.reset_index().astype('float64')
    return float64_data_df


def get_connected_vertices_filter(data_df, orig_data_df=None):
    if orig_data_df is None:
        connected_vertices_filter = data_df['general'] != 0
    else:
        connected_vertices_filter = data_df['general'] != 0
        timesteps = data_df['timestep'].unique()
        for t in timesteps:
            current_timestep_connected_vertices_filter = \
                orig_data_df.loc[orig_data_df['timestep'] == t]['general'] != 0
            prev_timestep_connected_vertices_filter = \
                orig_data_df.loc[
                    orig_data_df['timestep'] == t - 1
                ]['general'] != 0
            connected_vertices_filter.loc[data_df['timestep'] == t]['general'] = \
                current_timestep_connected_vertices_filter.reset_index().drop(
                    'index', axis=1) & \
                prev_timestep_connected_vertices_filter.reset_index().drop(
                    'index', axis=1)

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
        kwargs['bw_adjust'] = 1
        print(args, kwargs)
        p = sns.kdeplot(*args, **kwargs)
        p.figure.canvas.start_event_loop(sys.float_info.min)
    except ValueError:
        kwargs['bw_adjust'] = 5 * kwargs['bw_adjust']
        kwargs['color'] = 'red'
        print(args, kwargs)
        safe_kdeplot(*args, **kwargs)


def plot_pairs(
    data_df,
    output_path,
    data_folder_name,
    data_name,
    additional_name
):
    g = sns.PairGrid(data_df, vars=LABEL_TYPES, diag_sharey=False)
    g.map_upper(sns.histplot, bins=(100, 100))
    g.map_diag(sns.histplot, bins=100, kde=True, stat="probability",
               kde_kws={'bw_adjust': 0.25, 'cut': 0})
    g.map_lower(safe_kdeplot, cut=0, fill=True)
    tag = f"Centralities distributions in " \
        f"{data_folder_name}/{data_name}, {additional_name}"
    plot_name_to_save = create_pair_plots_name(
        output_path, data_folder_name, data_name, additional_name)
    process_plot(g, tag, plot_name_to_save)


def plot_histogram(data, idx_to_node_id, tag, out_file_name):
    data = data[~(data == -np.inf)]
    data = data[~(data == np.inf)]
    if "betweenness" in tag or "load" in tag:
        p = sns.displot(data, bins=100)
    else:
        p = sns.displot(data)
    p.set(title=tag)
    p.savefig(os.path.join(out_file_name, f"{tag}.png"))

    kde_tag = " ".join([tag, "kde"])
    if "betweenness" in tag or "load" in tag:
        pkde = sns.displot(data, kde=True, bins=100)
    else:
        pkde = sns.displot(data, kde=True)
    pkde.set(title=kde_tag)
    pkde.savefig(os.path.join(out_file_name, f"{kde_tag}.png"))


def plot_correlation(data_1, data_2, idx_to_node_id, tag, out_file_name):
    if "betweenness" in tag or "load" in tag:
        p = sns.displot(x=data_1, y=data_2, rug=True,
                        cbar=True, bins=(100, 100))
    else:
        p = sns.displot(x=data_1, y=data_2, rug=True, cbar=True)
    p.set(title=tag)
    p.set_axis_labels(x_var="target", y_var="baseline")
    p.savefig(os.path.join(out_file_name, f"{tag}.png"))


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


def create_centralities_correlations_single_plot(
    data_df,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    correlation_matrix = data_df[LABEL_TYPES].corr(method='spearman')
    p = sns.heatmap(correlation_matrix, annot=True, square=True,
                    vmin=0, vmax=1, cmap=sns.cm.rocket_r)
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=45)
    tag = f"Correlations between centralities in " \
        f"{data_folder_name}/{data_name} for {additional_name.replace('_', ' ')}"
    output_file_name = create_centralities_correlations_name(
        output_path, data_folder_name, data_name, additional_name)
    process_subplot(p, tag, output_file_name)


def create_centralities_correlations_plots(
    data_df,
    connected_vertices_filter,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    create_centralities_correlations_single_plot(
        data_df,
        output_path,
        data_folder_name,
        data_name,
        additional_name='_'.join(['all_vertices', additional_name]),
    )
    create_centralities_correlations_single_plot(
        data_df.loc[connected_vertices_filter],
        output_path,
        data_folder_name,
        data_name,
        additional_name='_'.join(['only_connected_vertices', additional_name])
    )


def get_time_diff_data_df(data_df, periods=1):
    time_data_df = data_df.set_index(['timestep', 'node_id'])
    time_diff_data_df = time_data_df.groupby(level=1).diff(
        periods=periods).reset_index().dropna()
    return time_diff_data_df


def plot_temporal_auto_correlation(data_df):
    pass  # TODO


def analyze_dataset(
    data_folder_name,
    data_name,
    output_path,
    run_complex_plots=False
):
    orig_data_df = load_labels(data_folder_name, data_name)
    data_df = convert_df_to_float64(orig_data_df)
    connected_vertices_filter = get_connected_vertices_filter(data_df)
    # logs_data_df = get_logs_data_df(data_df)

    diff_data_df = get_time_diff_data_df(data_df, periods=1)
    diff_connected_vertices_filter = get_connected_vertices_filter(
        diff_data_df, data_df)

    create_centralities_correlations_plots(
        data_df,
        connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name
    )

    create_centralities_correlations_plots(
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        "diff"
    )

    create_time_evolution_plots(
        data_df,
        connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        additional_name='log'
    )

    create_time_evolution_plots(
        data_df,
        connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name
    )

    create_time_evolution_plots(
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        additional_name='diff_log'
    )

    create_time_evolution_plots(
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        additional_name='diff'
    )

    if run_complex_plots:
        plot_pairs(data_df, output_path, data_folder_name,
                   data_name, "only connected vertices log")
        plot_pairs(data_df[connected_vertices_filter], output_path,
                   data_folder_name, data_name, "only connected vertices")
        # plot_pairs(logs_data_df, output_path, data_folder_name, data_name, "all vertices log")
        plot_pairs(data_df, output_path, data_folder_name,
                   data_name, "all vertices")
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
        "output_path", type=str,
        help="path of dir to save plots to")
    argparser.add_argument("--run-complex-plots", action='store_true')
    argparser.add_argument(
        "--matplotlib-backend", type=str, default="Agg",
        help="matplotlib backend to use, default is Agg, but supports TkAgg too.")

    args = argparser.parse_args()
    return args


def main():
    args = get_args()

    plotting_definitions(args.matplotlib_backend)

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
