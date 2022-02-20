# %%
import os
import sys
import errno
from argparse import ArgumentParser
import itertools
import pickle
import numpy as np
import networkx as nx
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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

RENAME_DATASETS = {
    "reality_mining_daily" : "Reality Mining Daily",
    "reality_mining_monthly" : "Reality Mining Monthly",
    "ia_retweet_pol" : "Ia-retweet-pol",
    "ia-slashdot-reply-dir" : "Ia-slashdot-reply-dir",
    "ia-digg-reply" : "Ia-digg-reply",
    "dnc_candidate_one" : "Email-DNC Candidate One",
    "dnc_candidate_two" : "Email-DNC Candidate Two",
    "ca_cit_hepph" : "Ca-cit-HepPh",
    "fb-wosn-friends_1" : "Fb-wosn-friends\_1",
    "fb-wosn-friends_2" : "Fb-wosn-friends\_2",
    "complab05" : "Complab05",
    "infocom05" : "Infocom05",
    "infocom06" : "Infocom06",
    "infocom06_hours" : "Infocom06\_hours",
    "infocom06_four_hours" : "Infocom06\_four\_hours",
    "intel05" : "Intel05",
}

RENAME_LABELS = {
    "betweenness_centrality": "Betweenness Centrality",
    "load_centrality": "Load Centrality",
    "k_core": "Core Centrality",
    "general": "Degree Centrality",
    "closeness_centrality": "Closeness Centrality",
    "page_rank": "PageRank Centrality",
}

LABEL_TYPES = list(RENAME_LABELS.values())


MULTI_HISTOGRAMS_COL_WRAP = 3

# %%


# %%

def plotting_definitions(matplotlib_backend):
    sns.set_context("paper")
    sns.set_style("ticks", {'font.family': 'serif',
                            'font.serif': 'Times New Roman'})
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = "Times New Roman"
    matplotlib.use(matplotlib_backend)


def load_labels(data_folder_name, data_name):
    with open(os.path.join(
        ".", "Pickles", data_folder_name, f"{data_name}_labels_df.pkl"
    ), "rb") as f:
        df = pickle.load(f)
    return df


def load_graphs(data_folder_name, data_name):
    with open(os.path.join(
            ".", "Pickles", data_folder_name, f"{data_name}.pkl"), 'rb') as f:
        graphs = pickle.load(f)
    return graphs

# %%


def plot_data_evolution(data_df, label_type, connected_vertices_filter):
    df_connected_vertices = data_df.loc[connected_vertices_filter]
    p = sns.lineplot(
        data=data_df,
        x='timestep',
        y=label_type,
        ci=95,
        label='all vertices'
    )
    p = sns.lineplot(
        data=df_connected_vertices,
        x='timestep',
        y=label_type,
        ci=95,
        label='only connected vertices'
    )
    return p


def first_letter_capital(string):
    if string:
        return f"{string[0].upper()}{string[1:]}"
    else:
        return ''


def capitalize_all_strings(axes):
    for label in ['xlabel', 'ylabel', 'zlabel']:
        try:
            curr_label = eval(f'axes.get_{label}()')
            eval(f'axes.set_{label}("{first_letter_capital(curr_label)}")')
        except AttributeError:
            pass
    legend = axes.get_legend()
    if legend is not None:
        for t in legend.texts:
            t.set_text(first_letter_capital(t.get_text()))
    axes.set_title(first_letter_capital(axes.get_title()))
    return


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
        f.write(tag.replace('_', r'\_'))
    if axes:
        capitalize_all_strings(axes)
    figure.tight_layout()
    figure.canvas.start_event_loop(sys.float_info.min)
    figure.savefig(plot_name_to_save, bbox_inches="tight")
    plt.close()


def process_subplot(p, tag, plot_name_to_save, x_label='', y_label='', z_label=''):
    process_figure(p.figure, p.axes, tag, plot_name_to_save,
                   x_label=x_label, y_label=y_label, z_label=z_label)


def process_plot(p, tag, plot_name_to_save, x_label='', y_label='', z_label=''):
    process_figure(p.fig, p.ax, tag, plot_name_to_save,
                   x_label=x_label, y_label=y_label, z_label=z_label)


def process_facet_grid(p, tag, plot_name_to_save):
    process_figure(p.fig, None, tag, plot_name_to_save)


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
    tag_label_type = label_type.replace(" ", "\ ")
    tag = f"{data_folder_name}/{data_name}" \
        f" {' '.join([additional_name, f'${tag_label_type}$'])} evolution over timesteps"
    process_subplot(p, tag, plot_name_to_save)


def get_log_guard(vector_to_guard, log_guard_scale=10):
    min_non_zero = vector_to_guard[~(vector_to_guard == 0)].min()
    log_guard = min_non_zero / log_guard_scale
    return log_guard


def guarded_log(arr, log_guard_scale=10):
    log_guard = get_log_guard(arr, log_guard_scale=log_guard_scale)
    return np.log10(arr + log_guard)


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
    occurance = np.log10(np.array(histograms).T + occurance_log_guard)
    surf = ax.plot_surface(
        histogram_bins,
        time_steps,
        occurance,
        linewidth=0,
        antialiased=True,
        cmap=sns.cm.rocket
    )

    tag_label_type = label_type.replace(" ", "\ ")
    tag = f"{data_folder_name}/{data_name} " \
        f"{' '.join([additional_name, f'${tag_label_type}$'])} 3D evolution over timesteps"
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
        x_label=f"{'sym' if is_diff else ''}log$_{{10}}$("
        f"{'diff ' if is_diff else ''}{label_type.replace('_', ' ')})",
        y_label="timestep",
        z_label="log$_{10}$(occurance)"
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
        # log_guard = get_log_guard(
        #     df_connected_vertices[label_type],
        #     log_guard_scale=10
        # )
        # hist_data_df[label_type] = np.log(hist_data_df[label_type] + log_guard)
        hist_data_df[label_type] = guarded_log(
            hist_data_df[label_type], log_guard_scale=10)
    all_hist, all_bins = np.histogram(
        hist_data_df[label_type],
        bins=10
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
    logv = np.abs(arr)*(np.power(10, shift))
    logv.loc[logv < 1.] = 1.
    logv = np.sign(arr)*np.log10(logv)
    return logv


def symlog(arr):
    min_abs_label = get_log_guard(np.abs(arr), log_guard_scale=2)
    label_center_log = -np.log10(min_abs_label)
    symlog_v = symlog_shift(arr, shift=label_center_log)
    symlog_v.loc[arr > 0] = symlog_v.loc[arr > 0] - label_center_log
    symlog_v.loc[arr < 0] = symlog_v.loc[arr < 0] + label_center_log
    return label_center_log, symlog_v


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


def create_label_histogram(
    data_df,
    connected_vertices_filter,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    log_scale = 'log' in additional_name
    diffs = 'diff' in additional_name
    kde = 'kde' in additional_name
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
    if kde:
        kind = 'kde'
    else:
        kind = 'hist'
    try:
        g = sns.displot(
            data_df_to_plot.loc[connected_vertices_filter],
            x=label_type,
            palette='viridis',
            kind=kind,
            log_scale=(log_scale and not diffs, log_scale and not diffs)
        )
    except np.linalg.LinAlgError:
        return

    if log_scale and diffs:
        plt.yscale('log')
        g.set_xlabels(f'symlog$_{{10}}$(diff {label_type})')
        trans = transforms.blended_transform_factory(
            g.ax.transData, g.ax.transAxes)

        plt.text(0, 0.95, '$-10^{%.2f}$ ~ $10^{%.2f}$' %
                 (-label_center_log, -label_center_log),
                 va='bottom', ha='center', transform=trans)
        plt.axvline(x=0, ymin=-0.1, ymax=1.1,
                    ls='--', lw=0.5, color='k')
    tag_label_type = label_type.replace(" ", "\ ")
    tag = f"{data_folder_name}/{data_name}" \
        f" {' '.join([additional_name, f'${tag_label_type}$'])} distribution"
    plot_name_to_save = create_data_distribution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        "_".join([label_type, 'histograms'])
    )
    process_plot(g, tag, plot_name_to_save)


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
        g.set_xlabels(f'symlog$_{{10}}$(diff {label_type})')
        trans = transforms.blended_transform_factory(
            g.ax.transData, g.ax.transAxes)
        plt.text(0, 0.95, '$-10^{%.2f}$~$10^{%.2f}$' %
                 (-label_center_log, -label_center_log),
                 va='bottom', ha='center', transform=trans)
        plt.axvline(x=0, ymin=-0.1, ymax=1.1,
                    ls='--', lw=0.5, color='k')
    tag_label_type = label_type.replace(" ", "\ ")
    tag = f"{data_folder_name}/{data_name}" \
        f" {' '.join([additional_name, f'${tag_label_type}$'])} evolution over timesteps"
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        additional_name,
        "_".join([label_type, 'histograms'])
    )
    process_plot(g, tag, plot_name_to_save)


def create_distribution_plot(
    data_df,
    connected_vertices_filter,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    for label_type in LABEL_TYPES:
        create_label_histogram(
            data_df,
            connected_vertices_filter,
            label_type,
            output_path,
            data_folder_name,
            data_name,
            additional_name
        )
        create_label_histogram(
            data_df,
            connected_vertices_filter,
            label_type,
            output_path,
            data_folder_name,
            data_name,
            '_'.join([additional_name, 'kde'])
        )


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
    output_file_name = os.path.join(
        base_output_path,
        rest_of_path.replace(' ', '_')
    )
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return output_file_name


def create_data_distribution_file_name(
    output_path,
    data_folder_name,
    data_name,
    additional_name,
    label_type
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"{additional_name}_{label_type}_distribution.png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
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


def create_dataset_info_file_name(
    output_path,
    data_folder_name,
    data_name
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"{data_name}_dataset_info.txt"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_largest_cc_evolution_file_name(
    output_path,
    data_folder_name,
    data_name
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        "largest_connected_component_evolution_over_time.png"
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


def create_temporal_autocorrelation_name(
    output_path,
    data_folder_name,
    data_name,
    label_type,
    additional_name=''
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"{label_type}_temporal_autocorrelations_{additional_name}.png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_pair_plots_name(
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    rest_of_path = os.path.join(
        data_folder_name,
        data_name,
        f"Centralities_distributions_{additional_name.replace(' ', '_')}.png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def convert_df_to_float32(data_df):
    columns_to_convert = {col: 'float32' for col in LABEL_TYPES}
    columns_to_convert['timestep'] = 'float32'
    try:
        float32_data_df = data_df.reset_index().astype(columns_to_convert)
    except ValueError:
        data_df['Degree Centrality'] = list(
            map(np.sum, data_df['Degree Centrality']))
        float32_data_df = data_df.reset_index().astype(columns_to_convert)
    return float32_data_df


def get_connected_vertices_filter(data_df, orig_data_df=None):
    if orig_data_df is None:
        connected_vertices_filter = data_df['Degree Centrality'] != 0
    else:
        connected_vertices_filter = data_df['Degree Centrality'] != 0
        timesteps = data_df['timestep'].unique()
        index_offset = len(orig_data_df.query(f"timestep == {timesteps[0]}"))
        for t in timesteps:
            current_timestep_vertices = orig_data_df.loc[
                orig_data_df['timestep'] == t
            ]
            current_timestep_connected_vertices_filter = current_timestep_vertices[
                'Degree Centrality'] != 0

            prev_timestep_vertices = orig_data_df.loc[
                orig_data_df['timestep'] == t - 1
            ]
            prev_timestep_connected_vertices_filter = prev_timestep_vertices[
                'Degree Centrality'] != 0

            connected_vertices_filter.loc[data_df['timestep'] == t] = current_timestep_connected_vertices_filter & \
                prev_timestep_connected_vertices_filter.rename(
                    index=lambda x: x + index_offset)

    return connected_vertices_filter


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
    additional_name=''
):
    g = sns.PairGrid(data_df, vars=LABEL_TYPES,
                     diag_sharey=False, corner=True, palette='viridis')
    g.map_lower(sns.histplot, bins=(100, 100))
    g.map_diag(sns.histplot, bins=20, log_scale=(False, True))
    tag = f"Centralities distributions in " \
        f"{data_folder_name}/{data_name}, diagonal in log y scale {additional_name}"
    plot_name_to_save = create_pair_plots_name(
        output_path, data_folder_name, data_name, additional_name)
    process_facet_grid(g, tag, plot_name_to_save)

    return


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


def mathify(string):
    math_space_string = string.replace(' ', '\\ ')
    return f'${math_space_string}$'


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
    p.set_yticklabels(
        [f'{mathify(t.get_text())}' for t in p.get_yticklabels()],
        rotation=45, va='top', ha='right')
    p.set_xticklabels(
        [f'{mathify(t.get_text())}' for t in p.get_xticklabels()],
        rotation=45, va='top', ha='right')
    tag = f"Correlations between centrality measures in " \
        f"{data_folder_name}/{data_name}, calculated on {additional_name.replace('_', ' ')}"
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
        additional_name='_'.join(['connected_vertices_only', additional_name])
    )


def get_time_diff_data_df(data_df, periods=1):
    time_data_df = data_df.set_index(['timestep', 'node_id'])
    time_diff_data_df = time_data_df.groupby(level=1).diff(
        periods=periods).reset_index().dropna()
    return time_diff_data_df


def get_log_data_df(data_df, log_guard_scale=10):
    log_data_df = data_df.copy()
    for label_type in LABEL_TYPES:
        log_data_df[label_type] = guarded_log(
            data_df[label_type], log_guard_scale=log_guard_scale)
    return log_data_df


def calc_temporal_correlations(data_df, only_connected=False):
    timesteps = data_df['timestep'].unique()
    time_data_df = data_df.set_index(['timestep', 'node_id'])
    if only_connected:
        time_data_df = time_data_df.query('`Degree Centrality` != 0')
    max_gap = len(timesteps) - 1
    correlations = []
    for gap in range(1, max_gap + 1):
        gap_correlations = []
        for timestep in timesteps[:-gap]:
            gap_correlations.append(time_data_df.loc[timestep].corrwith(
                time_data_df.loc[timestep+gap], method='spearman'))
        correlations.append(gap_correlations)
    gaps_corr_dfs_list = []
    min_timestep = timesteps.min()
    for i, corr, in enumerate(correlations):
        gap_corr_df = pd.DataFrame(corr).rename(
            index=lambda x: int(x + min_timestep)
        )
        gap_corr_df['time gap'] = i + 1
        gaps_corr_dfs_list.append(gap_corr_df)
    if gaps_corr_dfs_list:
        gaps_corr_df = pd.concat(gaps_corr_dfs_list).reset_index().rename(
            columns={"index": "timestep"}
        )
    else:
        gaps_corr_df = None
    return gaps_corr_df


def exponential_fit(x, y):
    pos_y_filter = y > 0
    if np.count_nonzero(pos_y_filter) < 2:
        return None
    naive_fit = np.polyfit(
        x.loc[pos_y_filter],
        np.log(y.loc[pos_y_filter]),
        1,
        w=np.sqrt(x.loc[pos_y_filter])
    )
    initial_guess = (np.exp(naive_fit[1]), naive_fit[0])
    try:
        naive_exp_fit = curve_fit(
            lambda t, a, b: a*np.exp(b*t),
            x.loc[pos_y_filter],
            y.loc[pos_y_filter],
            p0=initial_guess
        )
    except RuntimeError:
        naive_exp_fit = [initial_guess, None]
    # try:
    #     exp_fit = curve_fit(lambda t, a, b, c: a*np.exp(b*t) +
    #                         c, x, y, p0=(*naive_exp_fit[0], 0))
    # except RuntimeError:
    #     new_cov_mat = np.zeros([
    #         naive_exp_fit[0].shape[0] + 1,
    #         naive_exp_fit[0].shape[0] + 1
    #     ])
    #     new_cov_mat[:2, :2] = naive_exp_fit[1]
    #     exp_fit = [(*naive_exp_fit[0], 0), new_cov_mat]
    exp_fit = naive_exp_fit
    return exp_fit


def calc_labels_exp_fit(data_autocorrelations_df):
    timesteps = data_autocorrelations_df['timestep'].unique()
    fits = {}
    for label_type in LABEL_TYPES:
        label_fits = []
        for t in timesteps[:-1]:
            timed_data_autocorrelations_df = data_autocorrelations_df.query(
                f'timestep=={t}'
            )
            fit = None
            if np.all(~timed_data_autocorrelations_df[label_type].isnull()):
                fit = exponential_fit(
                    timed_data_autocorrelations_df['time gap'],
                    timed_data_autocorrelations_df[label_type]
                )
            label_fits.append(fit)
        fits[label_type] = label_fits
    return fits


def plot_temporal_auto_correlation_grid(
    temporal_correlations,
    label_type,
    x='time gap',
    col_hue='timestep',
    col_wrap=5,
    palette="crest",
    is_diff=False,
    is_log=False,
    only_connected=False
):
    g = sns.relplot(
        data=temporal_correlations,
        x=x, y=label_type, col=col_hue, hue=col_hue,
        kind="line", palette=palette, linewidth=4, zorder=5,
        col_wrap=col_wrap, height=2, aspect=1.5, legend=False, marker='o',
    )
    for ax_title, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.8, .85, f"{first_letter_capital(col_hue)}={ax_title}",
                transform=ax.transAxes, fontweight="bold")

        sns.lineplot(
            data=temporal_correlations, x=x, y=label_type,
            units=col_hue, estimator=None, color=".7", linewidth=1, ax=ax,
        )

    g.set_titles("")
    g.set_axis_labels("", "")

    g.fig.subplots_adjust(top=0.9)
    fig_title = r'Temporal auto spearman correlation of $\bf{' + \
        (r'diff}$ $\bf{in}$ $\bf{' if is_diff else "") + \
        f'{"log_{10}(" if is_log else ""}' + \
        label_type.replace(' ', r'\ ') + \
        f'{")" if is_log else ""}' + \
        r'}$ vs. $\bf{' + \
        x.replace(' ', r'\ ') + \
        rf'{"}"}$ by different {col_hue}s' + \
        f'{" calculated only on connected vertices" if only_connected else ""}'
    g.fig.suptitle(first_letter_capital(fig_title))

    return g


def plot_temporal_auto_correlation_averages(temporal_correlations, label_type):
    # average_temporal_correlations = temporal_correlations.groupby(
    #     ['time gap']
    # ).mean().reset_index()
    # new_data_df_list = []
    # for label in LABEL_TYPES:
    #     new_data_df = pd.DataFrame()
    #     new_data_df['label_value'] = temporal_correlations[label]
    #     new_data_df['label_type'] = f'{label} autocorrelation'
    #     for i in ['timestep', 'time gap']:
    #         new_data_df[i] = temporal_correlations[i]
    #     new_data_df_list.append(new_data_df)
    # new_data_df = pd.concat(new_data_df_list)
    # p = sns.scatterplot(
    #     data=new_data_df.query('timestep==0'), x='time gap', y='label_value',
    #     hue='label_type', style='label_type', palette='viridis')
    # new_data_df['label_type'] = new_data_df['label_type'] + ' average'
    # p = sns.lineplot(
    #     data=new_data_df, x='time gap', y='label_value',
    #     hue='label_type', palette="rocket", style='label_type', ci='sd')
    min_timestep = temporal_correlations['timestep'].min()
    p = sns.lineplot(
        data=temporal_correlations, x='time gap', y=label_type, marker='o',
        zorder=5, label=f'Average autocorrelation', ci='sd')
    p = sns.lineplot(
        data=temporal_correlations.query(f'timestep == {min_timestep}'),
        x='time gap', y=label_type, marker='s', color="C2", zorder=4, ax=p,
        label=f'Autocorrelation at timestep={min_timestep}')
    averages_df = temporal_correlations.groupby(
        'time gap').mean().reset_index()
    fit_range = range(len(averages_df['time gap']))
    exp_fit = exponential_fit(
        averages_df['time gap'].loc[fit_range],
        averages_df[label_type].loc[fit_range]
    )
    plot_exp_fit = exp_fit is not None
    if plot_exp_fit:
        averages_df[f'exponential fit {label_type}'] = exp_fit[0][0] * \
            np.exp(exp_fit[0][1]*averages_df['time gap'])
        p = sns.lineplot(
            data=averages_df, x='time gap', y=f'exponential fit {label_type}',
            zorder=7, color="C1",
            label=f'Exponential fit: $({exp_fit[0][0]:.1e}) \cdot e^{{{exp_fit[0][1]:.1e}}}$'
        )
        p.set(yscale='log')
    return p, plot_exp_fit


def plot_single_label_temporal_auto_correlation_grid(
    temporal_correlations,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    x,
    col_hue,
    additional_name=''
):
    is_diff = 'diff' in additional_name
    is_log = 'log' in additional_name
    only_connected = 'only_connected' in additional_name
    col_wrap = int(np.ceil(len(temporal_correlations[col_hue].unique())/6))
    g = plot_temporal_auto_correlation_grid(
        temporal_correlations,
        label_type,
        x=x,
        col_hue=col_hue,
        col_wrap=col_wrap,
        is_diff=is_diff,
        is_log=is_log,
        only_connected=only_connected
    )

    tag_label_type = label_type.replace(" ", "\ ")
    tag = f'Temporal auto spearman correlation of {"diff" if is_diff else ""} ' \
        f'{"log$_{10}$(" if is_log else ""}${tag_label_type}${")" if is_log else ""} vs. {x} ' \
        f'by different {col_hue}s{" calculated only on connected vertices" if only_connected else ""}'

    output_file_name = create_temporal_autocorrelation_name(
        output_path, data_folder_name, data_name, label_type,
        f"vs_{x}_by_{col_hue}_{additional_name}")

    process_facet_grid(g, tag, output_file_name)


def plot_single_label_temporal_auto_correlation_average(
    temporal_correlations,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    only_connected = 'only_connected' in additional_name
    g, log_scale = plot_temporal_auto_correlation_averages(
        temporal_correlations,
        label_type,
    )

    tag_label_type = label_type.replace(" ", "\ ")

    tag = f'Temporal auto spearman correlation of ${tag_label_type}$ vs. time gap {additional_name.replace("_", " ")}'

    output_file_name = create_temporal_autocorrelation_name(
        output_path, data_folder_name, data_name, label_type,
        f"vs_time_gap_averages_{additional_name}")

    process_subplot(g, tag, output_file_name,
                    y_label=f'{"Log$_{10}$(" if log_scale else ""}' +
                    f'autocorrelation of {label_type}' +
                    f'{")" if log_scale else ""}' +
                    f'{" only connected" if only_connected else ""}')


def plot_single_label_temporal_auto_correlations(
    temporal_correlations,
    label_type,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    plot_single_label_temporal_auto_correlation_average(
        temporal_correlations,
        label_type,
        output_path,
        data_folder_name,
        data_name,
        additional_name=additional_name
    )
    plot_single_label_temporal_auto_correlation_grid(
        temporal_correlations,
        label_type,
        output_path,
        data_folder_name,
        data_name,
        x='time gap',
        col_hue='timestep',
        additional_name=additional_name
    )
    plot_single_label_temporal_auto_correlation_grid(
        temporal_correlations,
        label_type,
        output_path,
        data_folder_name,
        data_name,
        x='timestep',
        col_hue='time gap',
        additional_name=additional_name
    )


def plot_temporal_auto_correlation(
    temporal_correlations,
    output_path,
    data_folder_name,
    data_name,
    additional_name=''
):
    if temporal_correlations is None:
        return
    for label_type in LABEL_TYPES:
        plot_single_label_temporal_auto_correlations(
            temporal_correlations,
            label_type,
            output_path,
            data_folder_name,
            data_name,
            additional_name=additional_name
        )
    return


def create_all_correlation_plots(
    data_df,
    connected_vertices_filter,
    diff_data_df,
    diff_connected_vertices_filter,
    output_path,
    data_folder_name,
    data_name
):
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
        additional_name="diff"
    )
    return


def create_all_distribution_plots(
    data_df,
    connected_vertices_filter,
    diff_data_df,
    diff_connected_vertices_filter,
    output_path,
    data_folder_name,
    data_name
):
    # create_distribution_plot(
    #     data_df,
    #     connected_vertices_filter,
    #     output_path,
    #     data_folder_name,
    #     data_name,
    #     additional_name=''
    # )

    create_distribution_plot(
        data_df,
        connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        additional_name='log'
    )

    # create_distribution_plot(
    #     diff_data_df,
    #     diff_connected_vertices_filter,
    #     output_path,
    #     data_folder_name,
    #     data_name,
    #     additional_name='diff'
    # )

    create_distribution_plot(
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        additional_name='diff_log'
    )


def create_all_time_evolution_plots(
    data_df,
    connected_vertices_filter,
    diff_data_df,
    diff_connected_vertices_filter,
    output_path,
    data_folder_name,
    data_name
):
    create_time_evolution_plots(
        data_df,
        connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name,
        additional_name=''
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


def plot_all_temporal_autocorrelations(
    data_autocorrs_df,
    diff_data_autocorrs_df,
    diff_log_data_autocorrs_df,
    data_autocorrs_only_connected_df,
    output_path,
    data_folder_name,
    data_name
):
    plot_temporal_auto_correlation(
        data_autocorrs_df,
        output_path,
        data_folder_name,
        data_name,
        additional_name=''
    )

    plot_temporal_auto_correlation(
        diff_data_autocorrs_df,
        output_path,
        data_folder_name,
        data_name,
        additional_name='diff'
    )

    plot_temporal_auto_correlation(
        diff_log_data_autocorrs_df,
        output_path,
        data_folder_name,
        data_name,
        additional_name='diff_log'
    )

    plot_temporal_auto_correlation(
        data_autocorrs_only_connected_df,
        output_path,
        data_folder_name,
        data_name,
        additional_name='only_connected'
    )
    return


def get_largest_cc_over_time_df(graphs):
    num_vertices, largest_cc_over_time = get_largest_connected_component_over_time(
        graphs)
    num_connected_vertices = get_number_of_connected_vertices_over_time(
        graphs
    )
    degrees = get_degrees_over_time(graphs)
    connected_size_df = pd.DataFrame(
        {
            "Largest CC": largest_cc_over_time,
            "timestep": range(len(graphs)),
            "Number of connected vertices": num_connected_vertices,
            "Degrees": degrees
        }
    )
    connected_size_df['Largest CC %'] = 100 * \
        connected_size_df['Largest CC'] / num_vertices
    return num_vertices, connected_size_df


def plot_largest_cc_over_time(
    dataset_graphs,
    data_folder_name,
    data_name,
    output_path
):
    num_vertices, largest_cc_over_time = get_largest_cc_over_time_df(
        dataset_graphs)

    g = sns.lineplot(data=largest_cc_over_time, x="timestep",
                     y="Number of connected vertices", color='C0',
                     label="number of connected vertices")
    g1 = sns.lineplot(data=largest_cc_over_time, x="timestep",
                      y="Largest CC", color='C1', ax=g.axes,
                      label="size of largest connected component")
    capitalize_all_strings(g1.axes)
    ax = g1.twinx()
    g2 = sns.lineplot(data=largest_cc_over_time, x="timestep",
                      y="Largest CC %", color='C2', ax=ax, linestyle=":",
                      label="relative size of largest connected component")
    h1, l1 = g1.get_legend_handles_labels()
    h2, l2 = g2.get_legend_handles_labels()
    g2.axes.legend(loc=1, handles=h1+h2, labels=l1+l2)
    g1.legend([], [])

    tag = f"Size of largest connected component of " \
        f"{data_folder_name}/{data_name} over time"

    largest_cc_over_time_filename = create_largest_cc_evolution_file_name(
        output_path,
        data_folder_name,
        data_name
    )

    process_subplot(g2, tag, largest_cc_over_time_filename)

    return largest_cc_over_time


def get_cc_sizes_over_time(graphs):
    graph = graphs[0]
    if not graph.is_directed():
        cc_collection_func = nx.connected_components
    else:
        cc_collection_func = nx.weakly_connected_components
    cc_sizes_over_time = [
        list(
            filter(
                lambda x: x != 1,
                map(len, cc_collection_func(g))
            )
        ) for g in graphs
    ]
    return cc_sizes_over_time


def plot_connected_components_size_distribution_over_time(
    dataset_graphs,
    data_folder_name,
    data_name,
    output_path
):
    cc_sizes_over_time = get_cc_sizes_over_time(dataset_graphs)
    all_hist, all_bins = np.histogram(
        np.concatenate(cc_sizes_over_time), bins=10)
    timesteps = np.arange(len(cc_sizes_over_time))
    histograms = [np.histogram(cc_sizes, bins=all_bins)[0]
                  for cc_sizes in cc_sizes_over_time]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    time_steps, histogram_bins = np.meshgrid(
        timesteps.astype('int'), all_bins[:-1])
    surf = ax.plot_surface(
        histogram_bins,
        time_steps,
        np.array(histograms).T,
        linewidth=0,
        antialiased=True,
        cmap=sns.cm.rocket
    )

    tag = f"{data_folder_name}/{data_name} " \
        "connected components size distribution 3D evolution over timesteps"
    plot_name_to_save = create_data_evolution_file_name(
        output_path,
        data_folder_name,
        data_name,
        "",
        "_".join(["connected_components", '3D', 'histograms'])
    )
    process_subplot(
        surf,
        tag,
        plot_name_to_save,
        x_label="size of connected component",
        y_label="timestep",
        z_label="occurance"
    )
    return


def save_timesteps_num_vertices(
    dataset_graphs,
    largest_cc_over_time,
    data_folder_name,
    data_name,
    output_path
):
    number_of_timesteps = len(dataset_graphs)
    number_of_vertices = len(dataset_graphs[0].nodes())
    connected_vertices_all_times = get_connected_vertices_all_times(dataset_graphs)
    mean_largest_cc_over_time = np.mean(largest_cc_over_time)
    average_size_of_largest_cc = mean_largest_cc_over_time['Largest CC']
    average_percentage_of_largest_cc = mean_largest_cc_over_time['Largest CC %']
    average_number_of_connected_vertices = mean_largest_cc_over_time['Number of connected vertices']
    average_percentage_of_connected_vertices = mean_largest_cc_over_time['Number of connected vertices'] * 100 / number_of_vertices
    average_degree = mean_largest_cc_over_time['Degrees']
    clean_data_folder_name = data_folder_name.replace('_', r'\_')
    clean_data_name = data_name.replace('_', r'\_')

    label = f'tab:{data_folder_name}/{data_name}_general_info'

    tag = rf"""\begin{{table}}
\begin{{center}}
 \begin{{tabular}}{{||c | c||}} 
 \hline
 \textbf{{Feature}} & \textbf{{Value}} \\ [0.5ex] 
 \hline\hline
  \# of timesteps & {number_of_timesteps} \\
 \hline
  \# of vertices & {number_of_vertices} \\
 \hline
  \# of vertices connected in all timesteps & {len(connected_vertices_all_times)} \\
 \hline
  Average degree per timestep & {average_degree:.2f} \\
 \hline
  Average \# of connected vertices & {average_number_of_connected_vertices:.1f} \\
 \hline
  Average percentage of connected vertices & {average_percentage_of_connected_vertices:.1f}\% \\
 \hline
  Average size of largest CC & {average_size_of_largest_cc:.1f} \\
 \hline
  Average percentage of largest CC & {average_percentage_of_largest_cc:.1f}\% \\
 \hline
\end{{tabular}}
\caption[{RENAME_DATASETS[data_name]} General Information]{{{clean_data_folder_name}/{clean_data_name} general information}}
\label{{{label}}}
\end{{center}}
\end{{table}}"""

    dataset_info_file_name = create_dataset_info_file_name(
        output_path, data_folder_name, data_name)

    with open(dataset_info_file_name, 'w') as f:
        f.write(tag)
    with open(dataset_info_file_name[:-4] + '_ref.txt', 'w') as f:
        f.write(rf'\ref{{{label}}}')


def analyze_graphs(
    dataset_graphs,
    data_folder_name,
    data_name,
    output_path
):
    plot_connected_components_size_distribution_over_time(
        dataset_graphs,
        data_folder_name,
        data_name,
        output_path
    )
    largest_cc_over_time = plot_largest_cc_over_time(
        dataset_graphs,
        data_folder_name,
        data_name,
        output_path
    )
    save_timesteps_num_vertices(
        dataset_graphs,
        largest_cc_over_time,
        data_folder_name,
        data_name,
        output_path
    )
    return


def analyze_dataset(
    data_df,
    dataset_graphs,
    data_folder_name,
    data_name,
    output_path,
    run_complex_plots=False
):
    # orig_data_df = load_labels(data_folder_name, data_name)
    # data_df = convert_df_to_float32(orig_data_df)
    analyze_graphs(
        dataset_graphs,
        data_folder_name,
        data_name,
        output_path
    )
    connected_vertices_filter = get_connected_vertices_filter(data_df)
    log_data_df = get_log_data_df(data_df)

    diff_data_df = get_time_diff_data_df(data_df, periods=1)
    diff_connected_vertices_filter = get_connected_vertices_filter(
        diff_data_df, data_df)

    diff_log_data_df = get_time_diff_data_df(log_data_df, periods=1)

    create_all_correlation_plots(
        data_df,
        connected_vertices_filter,
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name
    )

    create_all_distribution_plots(
        data_df,
        connected_vertices_filter,
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name
    )

    create_all_time_evolution_plots(
        data_df,
        connected_vertices_filter,
        diff_data_df,
        diff_connected_vertices_filter,
        output_path,
        data_folder_name,
        data_name
    )

    data_autocorrs_df = calc_temporal_correlations(data_df)
    diff_data_autocorrs_df = calc_temporal_correlations(diff_data_df)
    diff_log_data_autocorrs_df = calc_temporal_correlations(diff_log_data_df)
    data_autocorrs_only_connected_df = calc_temporal_correlations(
        data_df, only_connected=True)
    # data_exp_fits = calc_labels_exp_fit(data_autocorrs_df)

    plot_all_temporal_autocorrelations(
        data_autocorrs_df,
        diff_data_autocorrs_df,
        diff_log_data_autocorrs_df,
        data_autocorrs_only_connected_df,
        output_path,
        data_folder_name,
        data_name
    )

    if run_complex_plots:
        plot_pairs(data_df, output_path, data_folder_name,
                   data_name, "")
        plot_pairs(data_df[connected_vertices_filter], output_path,
                   data_folder_name, data_name, "only connected vertices")
        plot_pairs(log_data_df, output_path, data_folder_name,
                   data_name, "log scale")
        plot_pairs(log_data_df[connected_vertices_filter], output_path,
                   data_folder_name, data_name, "only connected vertices log scale")
        # plot_pairs(data_df, output_path, data_folder_name,
        #            data_name, "all vertices")
    return


def update_graphs(graphs):
    all_nodes_set = set()
    for g in graphs:
        all_nodes_set.update(g.nodes())

    for g in graphs:
        g.add_nodes_from(all_nodes_set)

    return graphs


def load_all_datasets():
    all_data_dfs_list = []
    for folder, datasets in DATASETS.items():
        for dataset in datasets:
            new_data_df = load_labels(folder, dataset)
            new_data_df['dataset'] = f"{folder}_{dataset}"
            new_data_df = new_data_df.rename(columns=RENAME_LABELS)
            new_data_df_float32 = convert_df_to_float32(new_data_df)
            dataset_graphs = load_graphs(folder, dataset)
            all_data_dfs_list.append(
                {
                    "folder": folder,
                    "dataset": dataset,
                    "data": new_data_df_float32,
                    "graphs": update_graphs(dataset_graphs)
                }
            )
    all_data_df = pd.concat([item["data"] for item in all_data_dfs_list])
    return all_data_dfs_list, all_data_df


def get_largest_connected_component_over_time(graphs):
    graph = graphs[0]
    if not graph.is_directed():
        largest_cc_over_time = [
            len(max(nx.connected_components(g), key=len)) for g in graphs
        ]
    else:
        largest_cc_over_time = [
            len(max(nx.weakly_connected_components(g), key=len)) for g in graphs
        ]
    number_of_nodes = len(graph.nodes())
    return number_of_nodes, largest_cc_over_time


def get_number_of_connected_vertices_over_time(graphs):
    return [len(g.nodes()) - len(list(nx.isolates(g))) for g in graphs]


def get_degrees_over_time(graphs):
    return [np.mean([d for _, d in g.degree()]) for g in graphs]


def get_connected_vertices_all_times(graphs):
    prev_graph = graphs[0]
    all_intersection = np.array([n for n, d in prev_graph.degree() if d > 0])
    for g in graphs[1:]:
        connected_nodes = np.array([n for n, d in g.degree() if d > 0])
        all_intersection = np.intersect1d(all_intersection, connected_nodes)
        prev_graph = g
    return all_intersection


def compare_all_datasets(output_path, run_complex_plots=False):
    all_data_dfs_list, all_data_df = load_all_datasets()
    all_connected_vertices_filter = get_connected_vertices_filter(all_data_df)
    create_centralities_correlations_plots(
        all_data_df,
        all_connected_vertices_filter,
        output_path, "all_datasets", "combined", ""
    )
    return all_data_dfs_list


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

    print("comparing all datasets")
    all_data_list = compare_all_datasets(
        args.output_path,
        run_complex_plots=args.run_complex_plots
    )

    for dataset_item in all_data_list:
        folder_name = dataset_item['folder']
        dataset_name = dataset_item['dataset']
        dataset_data = dataset_item['data'].drop(columns='dataset')
        dataset_graphs = dataset_item['graphs']
        print(f"analyzing {folder_name}/{dataset_name}")
        analyze_dataset(
            dataset_data,
            dataset_graphs,
            folder_name,
            dataset_name,
            args.output_path,
            run_complex_plots=args.run_complex_plots
        )
    # for folder, datasets in DATASETS.items():
    #     for dataset in datasets:
    #         print(f"analyzing {folder}/{dataset}")
    #         analyze_dataset(
    #             folder,
    #             dataset,
    #             args.output_path,
    #             run_complex_plots=args.run_complex_plots
    #         )
    # compare_all_datasets()
    return


if '__main__' == __name__:
    main()

# %%
