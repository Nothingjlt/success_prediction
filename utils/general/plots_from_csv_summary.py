import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from itertools import product
import os
import errno
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')


MODEL_TYPES = [
    "GCNRNN"
]
DATASETS = [
    "reality_mining_daily",
    "reality_mining_monthly",
    "ia_retweet_pol",
    "ia-slashdot-reply-dir",
    "ia-digg-reply",
    "dnc_candidate_one",
    "dnc_candidate_two",
    "ca_cit_hepph",
    "fb-wosn-friends_1",
    "fb-wosn-friends_2",
    "complab05",
    "infocom05",
    "infocom06",
    "infocom06_hours",
    "infocom06_four_hours",
    "intel05",
]
DATASETS_PRESENTABLE = {
    "reality_mining_daily": "$reality_mining/reality_mining_daily$".replace('_', '\_'),
    "reality_mining_monthly": "$reality_mining/reality_mining_monthly$".replace('_', '\_'),
    "ia_retweet_pol": "$ia-retweet-pol/ia_retweet_pol$".replace('_', '\_'),
    "ia-slashdot-reply-dir": "$ia-slashdot-reply-dir/ia-slashdot-reply-dir$".replace('_', '\_'),
    "ia-digg-reply": "$ia-digg-reply/ia-digg-reply$".replace('_', '\_'),
    "dnc_candidate_one": "$dnc/dnc_candidate_one$".replace('_', '\_'),
    "dnc_candidate_two": "$dnc/dnc_candidate_two$".replace('_', '\_'),
    "ca_cit_hepph": "$ca-cit-HepPh/ca_cit_hepph$".replace('_', '\_'),
    "fb-wosn-friends_1": "$fb-wosn-friends/fb-wosn-friends_1$".replace('_', '\_'),
    "fb-wosn-friends_2": "$fb-wosn-friends/fb-wosn-friends_2$".replace('_', '\_'),
    "complab05": "$cambridge_haggle/complab05$".replace('_', '\_'),
    "infocom05": "$cambridge_haggle/infocom05$".replace('_', '\_'),
    "infocom06": "$cambridge_haggle/infocom06$".replace('_', '\_'),
    "infocom06_hours": "$cambridge_haggle/infocom06_hours$".replace('_', '\_'),
    "infocom06_four_hours": "$cambridge_haggle/infocom06_four_hours$".replace('_', '\_'),
    "intel05": "$cambridge_haggle/intel05$".replace('_', '\_'),
    "all_datasets": "all datasets"
}
DATASETS_SHORT_CAPTION = {
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
    "all_datasets": "All Datasets"
}
MEASURES = [
    "betweenness_centrality",
    "load_centrality",
    "k_core",
    "general",
    "closeness_centrality",
    "page_rank"
]
MEASURES_PRESENTABLE = {
    "betweenness_centrality": "$Betweenness\ Centrality$",
    "load_centrality": "$Load\ Centrality$",
    "k_core": "$Core\ Centrality$",
    "general": "$Degree\ Centrality$",
    "closeness_centrality": "$Closeness\ Centrality$",
    "page_rank": "$PageRank\ Centrality$"
}
METRICS = [
    "mses",
    "accuracies",
    "correlations",
    "maes",
]
METRICS_PRESENTABLE = {
    "mses": "MSE",
    "accuracies": "$R^2$ Accuracy",
    "correlations": "Correlation",
    "maes": "MAE"
}
MODELS = [
    "model_test",
    "model_train",
    "model_worst_case"
]
MODELS_PRESENTABLE = {
    "model_test": "NETSCAPE on test set",
    "model_train": "NETSCAPE on training set",
    "model_worst_case": "NETSCAPE on worst case set"
}
COMPARISON_MODELS = [
    "null_model",
    # "null_diff_model",
    "uniform_average",
    "linear_weighted_average",
    "square_root_weighted_average",
    "uniform_periodic_average",
    "weighted_periodic_average",
    "polynomial_regression",
    "first_order_model",
]
COMPARISON_MODELS_PRESENTABLE = {
    "null_model": "Null Model",
    # "null_diff_model": "Null diff model",
    "uniform_average": "Uniform Average",
    "linear_weighted_average": "Linear Weighted Average",
    "square_root_weighted_average": "Square Root Weighted Average",
    "uniform_periodic_average": "Uniform Periodic Average",
    "weighted_periodic_average": "Weighted Periodic Average",
    "polynomial_regression": "Polynomial Regression",
    "first_order_model": "First Order Model",
}
NUMBER_OF_ITERATIONS = 30


def plotting_definitions(matplotlib_backend):
    sns.set_context("paper")
    sns.set_style("ticks", {'font.family': 'Times New Roman'})
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = "Times New Roman"
    matplotlib.use(matplotlib_backend)


def first_letter_capital(string):
    if string:
        return f"{string[0].upper()}{string[1:]}"
    else:
        return ''


def capitalize_all_strings(ax):
    for label in ['xlabel', 'ylabel', 'zlabel']:
        try:
            curr_label = eval(f'ax.get_{label}()')
            eval(f'ax.set_{label}("{first_letter_capital(curr_label)}")')
        except AttributeError:
            pass
    legend = ax.get_legend()
    if legend is not None:
        for t in legend.texts:
            t.set_text(first_letter_capital(t.get_text()))
    ax.set_title(first_letter_capital(ax.get_title()))
    return


def process_figure(
    p,
    figure,
    ax,
    tag,
    plot_name_to_save,
    suptitle='',
    x_label='',
    y_label='',
    z_label='',
    x_rotation=None,
    y_rotation=None,
    move_legend=True,
    legend_top=False
):
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if z_label:
        ax.set_zlabel(z_label)
    if suptitle:
        figure.suptitle(tag)
    if x_rotation is not None:
        ax.tick_params(axis='x', rotation=x_rotation)
    if y_rotation is not None:
        ax.tick_params(axis='y', rotation=y_rotation)
    with open(f"{plot_name_to_save[:-4]}_title.txt", 'w') as f:
        f.write(tag.replace('_', r'\_'))
    if ax:
        capitalize_all_strings(ax)
    if move_legend:

        if not legend_top:
            loc = "upper center"
            bbox_to_anchor = (.5, 0)
        else:
            loc = "lower center"
            bbox_to_anchor = (.5, 1)
        sns.move_legend(
            p, loc, shadow=True,
            bbox_to_anchor=bbox_to_anchor, ncol=3, title=None, frameon=False,
        )
    figure.tight_layout()
    figure.canvas.start_event_loop(sys.float_info.min)
    figure.savefig(plot_name_to_save, bbox_inches="tight")
    plt.close()


def process_subplot(
        p, tag, plot_name_to_save, suptitle='', x_label='', y_label='',
        z_label='', x_rotation=None, y_rotation=None, move_legend=True, legend_top=False):
    process_figure(p, p.figure, p.axes, tag, plot_name_to_save, suptitle=suptitle,
                   x_label=x_label, y_label=y_label, z_label=z_label,
                   x_rotation=x_rotation, y_rotation=y_rotation, move_legend=move_legend, legend_top=legend_top)


def process_plot(p, tag, plot_name_to_save, suptitle='', x_label='', y_label='',
                 z_label='', x_rotation=None, y_rotation=None, move_legend=True, legend_top=False):
    process_figure(p, p.fig, p.ax, tag, plot_name_to_save, suptitle=suptitle,
                   x_label=x_label, y_label=y_label, z_label=z_label,
                   x_rotation=x_rotation, y_rotation=y_rotation, move_legend=move_legend, legend_top=legend_top)


def process_facet_grid(p, tag, plot_name_to_save, suptitle='',
                       x_rotation=None, y_rotation=None, move_legend=True, legend_top=False):
    process_figure(p, p.fig, None, tag, plot_name_to_save, suptitle=suptitle,
                   x_rotation=x_rotation, y_rotation=y_rotation, move_legend=move_legend, legend_top=legend_top)


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


def get_model_keys(metric_type):
    models_presentable = dict(MODELS_PRESENTABLE)
    models_presentable.pop('model_worst_case')
    model_keys = list(models_presentable.values())
    comp_model_keys = [
        f'{comp_model} on test set' for comp_model in COMPARISON_MODELS_PRESENTABLE.values()]
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


def plot_model_dataset_measure_metric_df(
    df,
    model,
    dataset,
    measure,
    metric,
    hue_order,
    col_order,
    output_path
):
    with sns.plotting_context("paper", font_scale=2):
        p = sns.catplot(data=df, col='Comparison Model', col_order=col_order, col_wrap=3,
                        x=f'{METRICS_PRESENTABLE[metric]} score', kind='count',
                        hue=f'{METRICS_PRESENTABLE[metric]} score',
                        hue_order=hue_order, order=hue_order, legend=True,
                        legend_out=True, facet_kws={"legend_out": True}, dodge=False)
        p.add_legend()
        for ax in p.axes:
            split_title = ax.get_title().split(' ')
            comparison_model_score = split_title[3]
            comparison_model = comparison_model_score.replace(
                '_comparison_score', '')
            ax.set_title(
                f'{COMPARISON_MODELS_PRESENTABLE[comparison_model]} {METRICS_PRESENTABLE[metric]} score')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(left=False, bottom=False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            #  rotation=45, ha='right')
            # ax[0].set_title('')
        sns.despine(left=True)
        sns.move_legend(p, 'upper left', bbox_to_anchor=(
            0.7, 0.3), fontsize='large')

        tag = f'{METRICS_PRESENTABLE[metric]} performance of' +\
            f' {MODELS_PRESENTABLE[model]} for {MEASURES_PRESENTABLE[measure]}' +\
            f' in {DATASETS_PRESENTABLE[dataset]}'

        output_file_name = create_model_dataset_measure_metric_file_name(
            model,
            dataset,
            measure,
            metric,
            output_path
        )

        process_facet_grid(p, tag, output_file_name, move_legend=False)
    return


def plot_model_dataset_measure_heatmap_df(
    df,
    model,
    dataset,
    measure,
    hue_order,
    comparison_models_order,
    metrics_order,
    output_path
):
    with sns.plotting_context("paper", font_scale=1.5):
        cmap = sns.color_palette()[:len(hue_order)]
        # Order meant to match the order in cmap
        mymap = {s: i for s, i in zip(hue_order, range(len(hue_order)))}
        pivot_df = df.pivot(index='Comparison Model', columns='metric',
                            values='value').reindex(
                                comparison_models_order)[METRICS]
        num_pivot_df = pivot_df.applymap(
            lambda x: mymap.get(x) if x in mymap else x)
        p = sns.heatmap(num_pivot_df, square=True, linewidths=0.5, cmap=cmap, vmin=0, vmax=len(hue_order))
        colorbar = p.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([
            colorbar.vmin + r / len(cmap) * (0.5 + i) for i in range(len(cmap))
        ])
        colorbar.set_ticklabels(list(mymap.keys()))
        p.set(ylabel='', xlabel='')
        comparison_models = [l.get_text().replace(
            "_comparison_score", "") for l in p.get_yticklabels()]
        p.set_yticklabels([
            f'{COMPARISON_MODELS_PRESENTABLE[l]}' for l in comparison_models
        ])
        p.set_xticklabels(
            [f'{METRICS_PRESENTABLE[m.get_text()]}' for m in p.get_xticklabels()],
            rotation=45, ha='right', va='top'
        )

        tag = f'All metrics performance of {MODELS_PRESENTABLE[model]} for' +\
            f' {MEASURES_PRESENTABLE[measure]} in {DATASETS_PRESENTABLE[dataset]}'

        output_file_name = create_model_dataset_measure_heatmap_file_name(
            model,
            dataset,
            measure,
            output_path
        )

        process_subplot(p, tag, output_file_name, move_legend=False)
    return


def plot_model_dataset_measure_df(
    df,
    model,
    dataset,
    measure,
    hue_order,
    comparison_models_order,
    metrics_order,
    output_path
):
    # TODO change to heatmap
    with sns.plotting_context("paper", font_scale=3, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(data=df, row='Comparison Model', col='metric',
                        x=f'value', kind='count', hue=f'value',
                        row_order=comparison_models_order, col_order=metrics_order,
                        hue_order=hue_order, order=hue_order, legend=True,
                        legend_out=True, facet_kws={"legend_out": True}, dodge=False)
        p.add_legend()
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            comp_model_comparison_score = split_title[3]
            comp_model = comp_model_comparison_score.replace(
                '_comparison_score', '')
            ax[0].set_ylabel(f'{COMPARISON_MODELS_PRESENTABLE[comp_model]}',
                             rotation=45, ha='right')
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    curr_metric = split_title[-1]
                    subax.set(
                        xlabel=f'{METRICS_PRESENTABLE[curr_metric]} score')
                subax.set(title='')

        tag = f'All metrics performance of {MODELS_PRESENTABLE[model]} for' +\
            f' {MEASURES_PRESENTABLE[measure]} in {DATASETS_PRESENTABLE[dataset]}'

        output_file_name = create_model_dataset_measure_file_name(
            model,
            dataset,
            measure,
            output_path
        )

        process_facet_grid(p, tag, output_file_name)
    return


def plot_dataset_measure(dataset_measure_iter_df, dataset, measure, metric, metric_columns, comparison_results, output_path):
    # generate_two_dim_plot(
    #     dataset_measure_iter_df.T,
    #     dataset,
    #     measure,
    #     metric,
    #     metric_columns[0],
    #     metric_columns[-1]
    # )
    metric_columns.remove(f'model_worst_case_{metric}_0')
    list_of_metric_dfs_to_concat = [
        dataset_measure_iter_df.T[column] for column in metric_columns
    ]
    model_keys = get_model_keys(metric)
    dataset_measure_metric_df = pd.concat(
        list_of_metric_dfs_to_concat,
        keys=model_keys
    ).reset_index().rename(columns={0: METRICS_PRESENTABLE[metric], "level_0": "Model"})
    generate_plots(
        dataset_measure_metric_df,
        dataset,
        measure,
        METRICS_PRESENTABLE[metric],
        comparison_results,
        output_path
    )
    return


def clean_dataset_name(dataset):
    return dataset.replace('.out.csv', '')


def create_tag(dataset, measure, metric):
    return f"{DATASETS_PRESENTABLE[dataset]} {MEASURES_PRESENTABLE[measure]} {metric}"


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
        measure,
        "_".join([measure, metric]) + ".png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_model_dataset_measure_metric_file_name(
    model,
    dataset,
    measure,
    metric,
    output_path
):
    rest_of_path = os.path.join(
        "datasets",
        dataset,
        'metrics',
        "_".join([model, measure, metric]) + ".png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_model_dataset_measure_file_name(
    model,
    dataset,
    measure,
    output_path
):
    rest_of_path = os.path.join(
        "datasets",
        dataset,
        "_".join([model, measure, "all_metrics"]) + ".png"
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_model_dataset_measure_heatmap_file_name(
    model,
    dataset,
    measure,
    output_path
):
    rest_of_path = os.path.join(
        "datasets",
        dataset,
        "_".join([model, measure, "heatmap", "all_metrics"]) + ".png"
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
        'comparison_models',
        f'{model}_performance_in_{dataset}_compared_to_{comparison_model}.png'
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_measure_comp_model_filename(
        model, output_path, dataset='all_datasets'):
    rest_of_path = f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'performance_in_{dataset.replace("-", "_")}_compared_to_all_models.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_measure_split_by_measure_dataset_filename(
    model, metric, output_path):
    rest_of_path = f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'{metric}_performance_in_all_datasets_split_by_measures.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_measure_split_by_measure_dataset_stacked_filename(
    model, metric, output_path):
    rest_of_path = f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'{metric}_performance_in_all_datasets_split_by_measures_stacked.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_measure_split_by_measures_stacked_filename(
    model, metric, measure, output_path):
    rest_of_path = os.path.join(
        'all_datasets',
        metric,
        f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'{metric}_{measure}_performance_in_all_datasets_stacked.png'
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_measure_split_by_metric_comp_model_filename(
        model, output_path, dataset='all_datasets'):
    rest_of_path = f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'performance_in_{dataset.replace("-", "_")}_split_by_metric_compared_to_all_models.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_metric_measure_filename(
    model, output_path, dataset='all_datasets'
):
    rest_of_path = f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'performance_in_{dataset.replace("-", "_")}_by_measure_metric.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_total_model_comp_df_by_metric_measure_stacked_filename(
    model, output_path, dataset='all_datasets'
):
    rest_of_path = f'{MODELS_PRESENTABLE[model].replace(" ", "_")}_' +\
        f'performance_in_{dataset.replace("-", "_")}_by_measure_metric_stacked.png'
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def create_text_file_name(output_path, dataset, model, comp_model, measure):
    rest_of_path = os.path.join(
        f"{model}_comparison",
        "datasets",
        dataset,
        f'{model}_{measure}_performance_in_{dataset}_compared_to_{comp_model}.txt'
    )
    output_file_name = create_output_file_name(output_path, rest_of_path)
    return output_file_name


def plot_hist_and_kde_separately(df, x, hue, comparison_results, palette, is_metric_distance):
    d_min = df[x].min()
    d_max = df[x].max()
    log_scale = False
    if is_metric_distance:
        if not d_min == 0:
            d_min = np.log10(d_min)
            d_max = np.log10(d_max)
            log_scale = True

    df_copy = df.copy()
    df_copy['row'] = 'Histogram'
    df_copy_two = df_copy.copy()
    df_copy_two['row'] = 'KDE'

    all_df = pd.concat([df_copy, df_copy_two])

    g = sns.FacetGrid(data=all_df, row='row', sharey=False, sharex=True)

    for curve, color in palette.items():
        sns.kdeplot(
            data=df.query(f'Model=="{curve}"'),
            x=x,
            color=color,
            ax=g.axes_dict['KDE'],
            clip=(d_min, d_max),
            log_scale=log_scale,
        )
    hist_ax = g.axes_dict['Histogram']
    sns.histplot(
        data=df, x=x, hue=hue, palette=palette, bins=100, edgecolor='k',
        ax=hist_ax, log_scale=log_scale)
    old_legend = hist_ax.get_legend()
    handles = old_legend.legendHandles
    texts = []
    for t in old_legend.get_texts():
        open_style = ''
        close_style = ''
        inner_t = t.get_text()
        if t.get_text() in comparison_results:
            if comparison_results[inner_t] == 'Won':
                open_style = r'$\it{'
                close_style = '}$'
                inner_t = inner_t.replace(' ', '\ ')
            elif comparison_results[inner_t] == 'Lost':
                open_style = r'$\bf{'
                close_style = '}$'
                inner_t = inner_t.replace(' ', '\ ')
        texts.append(f'{open_style}{inner_t}{close_style}')

    g.add_legend(legend_data={t: h for h, t in zip(handles, texts)})

    old_legend.remove()
    if is_metric_distance:
        g.axes[0][0].invert_xaxis()
    return g


def plot_hist_and_kde_same_scale(df, x, hue, palette, is_metric_distance):
    if is_metric_distance:
        df_copy = df.copy()
        df_copy[x] = np.log10(df[x])
    else:
        df_copy = df
    p = sns.histplot(
        data=df_copy,
        x=x,
        hue=hue,
        palette=palette,
        bins=100,
        edgecolor='k'
    )
    ax = p.twinx()
    d_min = df_copy[x].min()
    d_max = df_copy[x].max()
    p2 = sns.kdeplot(
        data=df_copy,
        x=x,
        hue=hue,
        palette=palette,
        ax=ax,
        clip=(d_min, d_max),
        legend=False
    )
    p2.set_ylabel('')
    p2.set_yticks([])
    sns.despine()
    if is_metric_distance:
        p.figure.axes[0].invert_xaxis()
        p.set_xlabel(f'$log_{{10}}$({p.get_xlabel()})')

    return p


def generate_plots(dataset_measure_metric_df, dataset, measure, metric, comparison_results, output_path):
    model_names = dataset_measure_metric_df['Model'].loc[
        dataset_measure_metric_df['Model'].str.startswith(
            "NETSCAPE")].unique()
    non_model_names = dataset_measure_metric_df['Model'].loc[
        ~dataset_measure_metric_df['Model'].str.startswith(
            "NETSCAPE")].unique()
    # palette = {n: c for n, c in zip(
    #     model_names, sns.color_palette('dark:red', len(model_names)+1)[1:])}
    # npalette = {n: c for n, c in zip(non_model_names, sns.diverging_palette(
    #     150, 275, l=80, s=100, center="light", n=len(non_model_names)))}
    palette = {n: c for n, c in zip(
        model_names, sns.color_palette('Greens_r', len(model_names))[:])}
    npalette = {n: c for n, c in zip(non_model_names, sns.diverging_palette(
        290, 48, s=99, l=65, sep=1, n=len(non_model_names))[:])}
    palette.update(npalette)
    is_metric_distance = metric == "MAE" or metric == "MSE"
    p = plot_hist_and_kde_separately(
        dataset_measure_metric_df, metric, "Model", comparison_results, palette, is_metric_distance)
    # p = plot_hist_and_kde_same_scale(
    #     dataset_measure_metric_df, metric, "Model", palette, is_metric_distance)
    # if metric == "MAE" or metric == "MSE":
    #     fig.axes[0].invert_xaxis()
    tag = create_tag(dataset, measure, metric)
    output_file_name = create_dataset_measure_output_file_name(
        output_path,
        dataset,
        measure,
        metric
    )
    # process_figure(fig, fig, None, tag, output_file_name, move_legend=True)
    process_facet_grid(p, tag, output_file_name,
                       move_legend=True, legend_top=True)
    return


def plot_metric_df(model_metric_df, model, comparison_model, metric, hue_order, order, output_path):
    p = sns.catplot(
        data=model_metric_df,
        hue=f'{comparison_model}_comparison_score',
        x='measure',
        kind='count',
        edgecolor='k',
        hue_order=hue_order,
        order=order
    )
    tag = f'{MODELS_PRESENTABLE[model]} {METRICS_PRESENTABLE[metric]} performance against {COMPARISON_MODELS_PRESENTABLE[comparison_model]}'

    output_file_name = create_metric_df_output_file_name(
        output_path,
        model,
        comparison_model,
        metric
    )
    p.ax.set_xlabel('')
    xticklabels = p.axes[0][0].get_xticklabels()
    p.axes[0][0].set_xticklabels(
        [
            f'{MEASURES_PRESENTABLE[l.get_text()]}'
            for l in xticklabels
        ],
        rotation=45, ha="right")
    process_plot(p, tag, output_file_name)
    return


def plot_model_performance_df(model_performance_df, model, comparison_model, metric, hue_order, col_order, row_order, output_path, dodge=True, model_type="GCNRNN"):
    with sns.plotting_context("paper", font_scale=2.3, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(
            row='dataset',
            col='measure',
            hue=f'{comparison_model}_comparison_score',
            y=f'{comparison_model}_comparison_score',
            data=model_performance_df,
            kind='count',
            order=hue_order,
            hue_order=hue_order,
            col_order=col_order,
            row_order=[f'{r}_{model_type}' for r in row_order],
            edgecolor='k',
            dodge=dodge
        )
        # p.add_legend()
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            curr_dataset = split_title[2].replace(f"_{model_type}", "")
            ax[0].set_ylabel(DATASETS_PRESENTABLE[curr_dataset],
                             rotation=45, ha='right')
            ax[0].set_yticklabels(ax[0].get_yticklabels())
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    curr_measure = split_title[-1]
                    subax.set_xlabel(
                        f'{MEASURES_PRESENTABLE[curr_measure]} score count', rotation=45, ha='right')
                subax.set_title('')
        # p.set_ylabels('')
        # yticklabels = p.axes[0][0].get_yticklabels()
        # p.axes[0][0].set_yticklabels(
        #     [
        #         f'{DATASETS_PRESENTABLE[l.get_text().replace("_GCNRNN", "")]}'
        #         for l in yticklabels
        #     ],
        #     # rotation=45, ha='right'
        # )
        # for col_ax in p.axes[0]:
        #     split_title = col_ax.get_title().split(' ')
        #     measure = split_title[-1]
        #     col_ax.set_xlabel(
        #         f'{MEASURES_PRESENTABLE[measure]} score count', rotation=45, ha='right')
        #     col_ax.set_title('')
        tag = f'{model.replace("_", " ")} {metric} performance compared to {comparison_model.replace("_", " ")}'
        output_file_name = create_model_performance_df_file_name(
            output_path,
            model,
            comparison_model,
            metric
        )
        process_facet_grid(p, tag, output_file_name, legend_top=True)
    return


def plot_dataset_df(dataset_df, model, comparison_model, dataset, hue_order, col_order, row_order, output_path):
    with sns.plotting_context("paper", font_scale=2.3, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(
            col="measure",
            row="metric",
            x=f'{comparison_model}_comparison_score',
            data=dataset_df,
            hue=f'{comparison_model}_comparison_score',
            kind='count',
            order=hue_order,
            hue_order=hue_order,
            col_order=col_order,
            row_order=row_order,
            edgecolor='k',
            dodge=False
        )
        p.add_legend()
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            metric = split_title[2]
            ax[0].set_ylabel(f'{METRICS_PRESENTABLE[metric]} count',
                             rotation=90, va='bottom', ha='center')
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    measure = split_title[-1]
                    subax.set(xlabel=f'{MEASURES_PRESENTABLE[measure]} score')
                subax.set(title='')
        tag = f'{MODELS_PRESENTABLE[model]} performance in {DATASETS_PRESENTABLE[dataset]} compared to {COMPARISON_MODELS_PRESENTABLE[comparison_model]}'

        output_file_name = create_dataset_df_file_name(
            output_path,
            model,
            comparison_model,
            dataset
        )
        process_facet_grid(p, tag, output_file_name)
    return


def plot_total_model_comp_df_by_metric_measure(
    df,
    model,
    score_order,
    measure_order,
    metric_order,
    output_path,
    dataset='all_datasets'
):
    # TODO flip row/column, increase font size still not good.
    with sns.plotting_context("paper", font_scale=2.3, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(
            row="measure",
            col="metric",
            x='score',
            data=df,
            hue='score',
            kind='count',
            order=score_order,
            hue_order=score_order,
            row_order=measure_order,
            col_order=metric_order,
            edgecolor='k',
            dodge=False
        )
        p.add_legend()
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            curr_measure = split_title[2]
            ax[0].set_ylabel(f'{MEASURES_PRESENTABLE[curr_measure]} score',
                             rotation=45, ha='right')
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    curr_metric = split_title[-1]
                    subax.set_xlabel(
                        f'{METRICS_PRESENTABLE[curr_metric]} count', rotation=45, ha='right')
                subax.set(title='')
        tag = f'{MODELS_PRESENTABLE[model]} performance in {DATASETS_PRESENTABLE[dataset]} by centrality measure and evaluation metric'

        output_file_name = create_total_model_comp_df_by_metric_measure_filename(
            model,
            output_path,
            dataset
        )
        process_facet_grid(p, tag, output_file_name)
    return


def plot_total_model_comp_df_by_metric_measure_stacked(
    df,
    model,
    score_order,
    measure_order,
    metric_order,
    output_path,
    dataset='all_datasets'
):
    # TODO flip row/column, increase font size still not good.
    with sns.plotting_context("paper", font_scale=3, rc={'legend.fontsize': 'large'}):
        p = sns.displot(
            col="measure",
            x="metric",
            data=df.dropna(),
            hue='score',
            kind='hist',
            hue_order=score_order,
            col_order=measure_order,
            edgecolor='k',
            multiple='stack',
            alpha=1,
            height=7,
            aspect=0.6
        )
        # p.add_legend()
        for ax in p.axes:            
            for subax in ax:
                split_title = subax.get_title().split(' ')
                curr_measure = split_title[2]
                subax.set_xlabel(f'{MEASURES_PRESENTABLE[curr_measure]}')
                subax.set_xticklabels(
                    map(
                        lambda x: METRICS_PRESENTABLE[x.get_text()],
                        subax.get_xticklabels()
                    ), rotation=45, ha='right'
                )
                subax.set(title='')
        tag = f'{MODELS_PRESENTABLE[model]} performance in {DATASETS_PRESENTABLE[dataset]} by centrality measure and evaluation metric'

        output_file_name = create_total_model_comp_df_by_metric_measure_stacked_filename(
            model,
            output_path,
            dataset
        )
        process_facet_grid(p, tag, output_file_name)
    return


def plot_total_model_comp_df_by_measure_comp_model(
    comp_df,
    model,
    score_order,
    measure_order,
    comparison_models_order,
    output_path,
    dataset='all_datasets'
):
    with sns.plotting_context("paper", font_scale=3, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(
            col='measure',
            row="Comparison Model",
            x="score",
            data=comp_df,
            hue='score',
            kind='count',
            # dodge=False,
            order=score_order,
            hue_order=score_order,
            col_order=measure_order,
            row_order=comparison_models_order,
            legend=True,
            legend_out=True,
            facet_kws={"legend_out": True},
            edgecolor='k'
        )
        p.add_legend()
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            comp_model = split_title[3]
            ax[0].set_ylabel(COMPARISON_MODELS_PRESENTABLE[comp_model],
                             rotation=45, ha='right')
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    measure = split_title[-1]
                    subax.set_xlabel(
                        f'{MEASURES_PRESENTABLE[measure]} score',
                        rotation=45, ha='right'
                    )
                subax.set(title='')
        tag = f'{MODELS_PRESENTABLE[model]} performance in {DATASETS_PRESENTABLE[dataset]} compared to all models'
        output_file_name = create_total_model_comp_df_by_measure_comp_model_filename(
            model,
            output_path,
            dataset=dataset
        )
        process_facet_grid(p, tag, output_file_name)
    return


def plot_total_model_comp_df_by_measure_split_by_measure_dataset(
    comp_df,
    model,
    metric,
    score_order,
    measure_order,
    dataset_order,
    output_path
):
    with sns.plotting_context("paper", font_scale=4, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(
            col='measure',
            row="dataset",
            x="metric",
            data=comp_df.query(f'metric=="{metric}"'),
            hue='score',
            kind='count',
            hue_order=score_order,
            col_order=measure_order,
            row_order=dataset_order,
            legend=True,
            legend_out=True,
            facet_kws={"legend_out": True},
            edgecolor='k',
            height=5/2,
            aspect=2
        )
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            curr_dataset = split_title[2]
            curr_dataset_clean = curr_dataset.replace('_GCNRNN', '')
            curr_dataset_ylabel = DATASETS_SHORT_CAPTION[curr_dataset_clean].replace(' ', '\ ')
            ax[0].set_ylabel(f'${curr_dataset_ylabel}$',
                             rotation=45, ha='right')
            ax[0].set_yticklabels([])
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    measure = split_title[-1]
                    subax.set_xlabel(
                        f'{MEASURES_PRESENTABLE[measure]} score',
                        rotation=45, ha='right'
                    )
                subax.set(title='')
        subax.set_xticklabels([])
        tag = f'{MODELS_PRESENTABLE[model]} {METRICS_PRESENTABLE[metric]} performance in all datasets compared to all models, split by centrality measure'
        output_file_name = create_total_model_comp_df_by_measure_split_by_measure_dataset_filename(
            model,
            metric,
            output_path
        )
        process_facet_grid(p, tag, output_file_name, legend_top=True)
    return


def plot_total_model_comp_df_by_measure_split_by_measure_dataset_stacked(
    comp_df,
    model,
    metric,
    score_order,
    measure_order,
    dataset_order,
    output_path
):
    with sns.plotting_context("paper", font_scale=1, rc={'legend.fontsize': 'large'}):
        for measure in measure_order:
            p = sns.displot(
                x="dataset",
                data=comp_df.query(f'metric=="{metric}" & measure=="{measure}"'),
                hue='score',
                kind='hist',
                multiple='stack',
                hue_order=score_order,
                edgecolor='k',
            )
            p.ax.set_xlabel(MEASURES_PRESENTABLE[measure])
            p.ax.set_xticklabels(
                    map(
                        lambda x: DATASETS_SHORT_CAPTION[x.get_text().replace('_GCNRNN', '')].replace('\_', '_'),
                        p.ax.get_xticklabels()
                    ), rotation=90, ha='center'
                )
            tag = f'{MODELS_PRESENTABLE[model]} {METRICS_PRESENTABLE[metric]} performance in all datasets compared to all models in {MEASURES_PRESENTABLE[measure]} prediction'
            output_file_name = create_total_model_comp_df_by_measure_split_by_measures_stacked_filename(
                model,
                metric,
                measure,
                output_path
            )
            process_plot(p, tag, output_file_name)
        # p = sns.displot(
        #     col='measure',
        #     x="dataset",
        #     data=comp_df.query(f'metric=="{metric}"'),
        #     hue='score',
        #     kind='hist',
        #     multiple='stack',
        #     hue_order=score_order,
        #     col_order=measure_order,
        #     row_order=dataset_order,
        #     edgecolor='k',
        # )
        # for ax in p.axes:
        #     for subax in ax:
        #         split_title = subax.get_title().split(' ')
        #         curr_measure = split_title[2]
        #         subax.set_xlabel(MEASURES_PRESENTABLE[curr_measure])
        #         subax.set(title='')
        #         subax.set_xticklabels(
        #             map(
        #                 lambda x: DATASETS_SHORT_CAPTION[x.get_text().replace('_GCNRNN', '')].replace('\_', '_'),
        #                 subax.get_xticklabels()
        #             ), rotation=90, ha='center'
        #         )
        # # subax.set_xticklabels([])
        # tag = f'{MODELS_PRESENTABLE[model]} {METRICS_PRESENTABLE[metric]} performance in all datasets compared to all models, split by centrality measure'
        # output_file_name = create_total_model_comp_df_by_measure_split_by_measure_dataset_stacked_filename(
        #     model,
        #     metric,
        #     output_path
        # )
        # process_facet_grid(p, tag, output_file_name, legend_top=True)
    return


def plot_total_model_comp_df_by_measure_split_by_metric_comp_model(
    comp_df,
    model,
    metric_order,
    score_order,
    measure_order,
    comparison_models_order,
    output_path,
    dataset='all_datasets'
):
    with sns.plotting_context("paper", font_scale=4, rc={'legend.fontsize': 'large'}):
        p = sns.catplot(
            col='measure',
            row="Comparison Model",
            x="metric",
            data=comp_df,
            hue='score',
            kind='count',
            dodge=False,
            order=metric_order,
            hue_order=score_order,
            col_order=measure_order,
            row_order=comparison_models_order,
            legend=True,
            legend_out=True,
            facet_kws={"legend_out": True},
            edgecolor='k'
        )
        for ax in p.axes:
            split_title = ax[0].get_title().split(' ')
            comp_model = split_title[3]
            ax[0].set_ylabel(COMPARISON_MODELS_PRESENTABLE[comp_model],
                             rotation=45, ha='right')
            ax[0].set_yticklabels([])
            for subax in ax:
                if subax.get_xlabel():
                    split_title = subax.get_title().split(' ')
                    measure = split_title[-1]
                    subax.set_xlabel(
                        f'{MEASURES_PRESENTABLE[measure]} score',
                        rotation=45, ha='right'
                    )
                    subax.set_xticklabels(
                        subax.get_xticklabels(), rotation=45, ha='right')
                subax.set(title='')
        subax.set_xticklabels(
            map(
                lambda x: METRICS_PRESENTABLE[x.get_text()],
                subax.get_xticklabels()
            ), rotation=45
        )
        tag = f'{MODELS_PRESENTABLE[model]} performance in {DATASETS_PRESENTABLE[dataset]} compared to all models, split by evaluation metric'
        output_file_name = create_total_model_comp_df_by_measure_split_by_metric_comp_model_filename(
            model,
            output_path,
            dataset=dataset
        )
        process_facet_grid(p, tag, output_file_name, legend_top=True)
    return


def plot_total_model_comp_df(
        df, score_order, measure_order, comparison_models_order, metric_order,
        output_path, dataset='all_datasets'):
    for model in MODELS:
        list_of_comp_models_to_concat = []
        model_comparison_names = []
        for comparison_model in COMPARISON_MODELS:
            comp_model_score = f'{comparison_model}_comparison_score'
            sliced_df = df.query(f'index.str.contains("{model}")')[
                [comp_model_score, 'measure', 'metric', 'dataset']
            ].rename(columns={comp_model_score: "score"})
            list_of_comp_models_to_concat.append(sliced_df)
            model_comparison_names.append(comparison_model)
        comp_df = pd.concat(
            list_of_comp_models_to_concat, keys=model_comparison_names
        ).reset_index(
        ).rename(
            columns={0: 'score', "level_0": "Comparison Model"}
        )
        if dataset == 'all_datasets':
            for metric in METRICS:
                plot_total_model_comp_df_by_measure_split_by_measure_dataset_stacked(
                    comp_df, model, metric, score_order, measure_order,
                    [f'{d}_GCNRNN' for d in DATASETS], output_path
                )
                plot_total_model_comp_df_by_measure_split_by_measure_dataset(
                    comp_df, model, metric, score_order, measure_order,
                    [f'{d}_GCNRNN' for d in DATASETS], output_path
                )
        plot_total_model_comp_df_by_metric_measure_stacked(
            comp_df, model, score_order, measure_order, metric_order, output_path,
            dataset=dataset
        )
        plot_total_model_comp_df_by_measure_comp_model(
            comp_df, model, score_order, measure_order, comparison_models_order,
            output_path, dataset=dataset)
        plot_total_model_comp_df_by_metric_measure(
            comp_df, model, score_order, measure_order, metric_order, output_path,
            dataset=dataset
        )
        plot_total_model_comp_df_by_measure_split_by_metric_comp_model(
            comp_df, model, metric_order, score_order, measure_order,
            comparison_models_order, output_path, dataset=dataset
        )
    return


def prettify_number(num, exp_digits=1, precision=1, check_length=True):
    if not check_length:
        scientific = np.format_float_scientific(
            num, exp_digits=exp_digits, precision=precision)
        scientific = scientific.replace('e+', 'e')
        if precision == 0:
            scientific = scientific.replace('.', '')
        return scientific.replace('e', ' \\times 10^{')+"}"
    else:
        float_repr = f'{num:.2f}'
        if len(float_repr) > 9:
            return prettify_number(num, exp_digits=exp_digits, precision=precision, check_length=False)
        else:
            return float_repr


def get_last_comp_model_index(df):
    latest_index = 0
    latest_row = ''
    for row in df.index.values:
        num = int(row.split('_')[-1])
        if num >= latest_index:
            latest_index = num
            latest_row = row
    return latest_row


def prepare_metric_line(metric, comp_model, model_perf, comp_model_perf):
    model_metric_perf = model_perf.query(f'metric=="{metric}"')
    assert(len(model_metric_perf) == 1)
    model_avg_metric = model_metric_perf['average'][0]
    model_std_metric = model_metric_perf['std'][0]
    model_metric_won = model_metric_perf[
        f'{comp_model}_comparison_score'][0] == 'Won'
    model_metric_lost = model_metric_perf[
        f'{comp_model}_comparison_score'][0] == 'Lost'
    model_pvalue = model_metric_perf[f'{comp_model}_comparison_wilcoxon_pvalue'][0]
    model_perc_improve = model_metric_perf[
        f'{comp_model}_comparison_perc_improvement'][0]
    comp_model_metric_perf = comp_model_perf.query(f'metric == "{metric}"')
    comp_index = get_last_comp_model_index(comp_model_metric_perf)
    comp_model_avg_metric = comp_model_metric_perf['average'][comp_index]
    comp_model_std_metric = comp_model_metric_perf['std'][comp_index]
    # model_stats = rf'{model_avg_metric:.2f} \pm {model_std_metric:.2f}'
    # comp_model_stats = rf'{comp_model_avg_metric:.2f} \pm {comp_model_std_metric:.2f}'
    model_stats = rf'{prettify_number(model_avg_metric)} \pm {prettify_number(model_std_metric)}'
    comp_model_stats = rf'{prettify_number(comp_model_avg_metric)} \pm {prettify_number(comp_model_std_metric)}'
    if model_metric_won:
        model_stats = rf'\mathbf{{{model_stats}}}'
    if model_metric_lost:
        comp_model_stats = rf'\mathbf{{{comp_model_stats}}}'
    metric_line_list = [
        rf'\textbf{{{METRICS_PRESENTABLE[metric]}}}',
        f'${model_stats}$',
        rf'${comp_model_stats}$',
        f'${model_perc_improve:.2f}$\%',
        f'${prettify_number(model_pvalue, precision=0, check_length=False)}$'
    ]
    return metric_line_list


def save_model_comp_text_to_file(
    output_path,
    dataset,
    model,
    comp_model,
    measure,
    model_perf,
    comp_model_perf


):
    output_filename = create_text_file_name(
        output_path, dataset, model, comp_model, measure)

    metric_lines = []
    for metric in model_perf['metric']:
        metric_lines.append(prepare_metric_line(
            metric, comp_model, model_perf, comp_model_perf))

    table_headers = [
        r'\textbf{Metric}',
        'NESTCAPE Model',
        f'{COMPARISON_MODELS_PRESENTABLE[comp_model]}',
        r'Improvement (\%)',
        'P-value'
    ]

    metric_lines_joint = '\\\\\n\\hline\n        '.join(
        ['&'.join(l) for l in metric_lines])

    label = f'tab:{dataset}_{model}_{comp_model}_results'
    short_caption = f'{DATASETS_SHORT_CAPTION[dataset]} NETSCAPE Performance Compared to {COMPARISON_MODELS_PRESENTABLE[comp_model]}'
    caption = rf'NETSCAPE {MEASURES_PRESENTABLE[measure]} prediction results in {DATASETS_PRESENTABLE[dataset]}, compared to {COMPARISON_MODELS_PRESENTABLE[comp_model]}'

    text_to_save = rf"""\begin{{table}}[H]
\begin{{center}}
    \begin{{tabular}}{{||{' c |'*len(table_headers)}|}} 
        \hline
        {'&'.join(table_headers)}\\ [0.5ex] 
        \hline\hline
        {metric_lines_joint}\\
        \hline
    \end{{tabular}}
    \caption[{short_caption}]{{{caption}}}
    \label{{{label}}}
    \end{{center}}
\end{{table}}"""

    with open(output_filename, 'w') as f:
        f.write(text_to_save)
    with open(output_filename[:-4] + "_label.txt", 'w') as f:
        f.write(label)
    with open(output_filename[:-4] + "_ref.txt", 'w') as f:
        f.write(rf'\ref{{{label}}}')

    return


def get_model_comp_model_perf(df, dataset, model_type, model, measure, comp_model):
    model_perf = df.query(
        f'dataset=="{dataset}_{model_type}" & model=="{model}" &'
        f' metric != "losses" & measure=="{measure}"'
    )[
        [
            f'{comp_model}_comparison_wilcoxon_pvalue',
            f'{comp_model}_comparison_perc_improvement',
            f'{comp_model}_comparison_score',
            'metric',
            'average',
            'std'
        ]
    ]
    comp_model_perf = df.query(
        f'dataset=="{dataset}_{model_type}" &'
        f' model=="{comp_model}" &'
        f' metric != "losses" & measure=="{measure}"'
    )[['average', 'std', 'metric']]
    return model_perf, comp_model_perf


def save_model_comparison_textual(df, num_of_iterations, output_path, model_type):
    averages = df[[str(i) for i in range(num_of_iterations)]].mean(axis=1)
    stds = df[[str(i) for i in range(num_of_iterations)]].std(axis=1)
    df['average'] = averages
    df['std'] = stds
    for measure, dataset, model, comp_model in product(
            MEASURES, DATASETS, MODELS, COMPARISON_MODELS):
        model_perf, comp_model_perf = get_model_comp_model_perf(
            df, dataset, model_type, model, measure, comp_model)
        save_model_comp_text_to_file(
            output_path,
            dataset,
            model,
            comp_model,
            measure,
            model_perf,
            comp_model_perf
        )
    return


def prepare_args():
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
    argparser.add_argument(
        "--matplotlib-backend", type=str, default="Agg",
        help="matplotlib backend to use for creating the plots"
    )
    argparser.add_argument(
        '--skip-textuals', action='store_true',
        help="Decide whether to skip the printing of textual format results"
    )
    argparser.add_argument(
        '--dataset', type=str, default="all_datasets", choices=DATASETS+['all_datasets'],
        help="The dataset to plot for. Default to all_datasets"
    )
    argparser.add_argument(
        '--measure', type=str, default="all_measures", choices=MEASURES+['all_measures'],
        help="The measure to plot for. Default to all_measures"
    )

    args = argparser.parse_args()
    return args


def main():
    args = prepare_args()

    plotting_definitions(args.matplotlib_backend)

    df = pd.read_csv(args.file_to_parse, index_col=0)
    # df = df.rename(columns={"Unnamed: 0": "model_metric"})

    df = df.mask(df == 'won', 'Won')
    df = df.mask(df == 'lost', 'Lost')
    df = df.mask(df == 'inconclusive', 'Inconclusive')

    if not args.dataset == 'all_datasets':
        global DATASETS
        DATASETS = [args.dataset]

    if not args.measure == 'all_measures':
        global MEASURES
        MEASURES = [args.measure]

    if args.skip_textuals:
        print('Skipping saving of textual results')
    else:
        print('Saving textual results')
        textual_output_folder = os.path.join(
            args.output_path,
            'texts'
        )
        save_model_comparison_textual(
            df, args.num_of_iterations, textual_output_folder, args.model_type)

    hue_order = ["Won", "Lost", "Inconclusive"]
    hues_cmap = sns.color_palette()[:3]

    print('Plotting total model comparisons')
    plot_total_model_comp_df(df, hue_order, MEASURES,
                             COMPARISON_MODELS, METRICS, args.output_path)

    for model in MODELS:
        print(f"plotting {model} model")
        model_folder_path = os.path.join(
            args.output_path,
            f"{model}_comparison"
        )
        model_performance_df = df.query(f'model=="{model}"')
        for comparison_model in COMPARISON_MODELS:
            print(f"\tplotting {comparison_model} comparison model")
            plot_model_performance_df(
                model_performance_df,
                model,
                comparison_model,
                "all metrics",
                hue_order,
                MEASURES,
                DATASETS,
                os.path.join(
                    model_folder_path,
                    "comparisons"
                )
            )
            for metric in METRICS:
                print(f"\t\tplotting {metric} metric")
                # metric_df = get_metric_df(df, metric)
                metric_performance_df = model_performance_df.query(
                    f'metric=="{metric}"'
                )

                plot_model_performance_df(
                    metric_performance_df,
                    model,
                    comparison_model,
                    metric,
                    hue_order,
                    MEASURES,
                    DATASETS,
                    os.path.join(
                        model_folder_path,
                        "comparisons",
                        f"compared_vs_{comparison_model}"
                    ),
                    dodge=False
                )
                plot_metric_df(
                    metric_performance_df,
                    model,
                    comparison_model,
                    metric,
                    hue_order,
                    MEASURES,
                    os.path.join(model_folder_path, metric)
                )

    for dataset in DATASETS:
        print(f"plotting {dataset} dataset")
        dataset_df = df.query(f'dataset=="{dataset}_{args.model_type}"')
        dataset_folder_path = os.path.join(
            args.output_path,
            dataset
        )
        plot_total_model_comp_df(
            dataset_df,
            hue_order,
            MEASURES,
            COMPARISON_MODELS,
            METRICS,
            dataset_folder_path,
            dataset=dataset
        )
        for model in MODELS:
            print(f"\tplotting {model} model")
            model_folder_path = os.path.join(
                args.output_path,
                f"{model}_comparison"
            )
            dataset_model_df = dataset_df.query(f'model=="{model}"')
            for comparison_model in COMPARISON_MODELS:
                print(f"\t\tplotting {comparison_model} comparison model")
                plot_dataset_df(
                    dataset_model_df,
                    model,
                    comparison_model,
                    dataset,
                    hue_order,
                    MEASURES,
                    METRICS,
                    model_folder_path
                )
            for measure in MEASURES:
                print(f"\t\tPlotting {measure} measure")
                dataset_measure_model_df = \
                    dataset_df.query(
                        f'measure=="{measure}" & model=="{model}"')
                dataset_measure_model_melt_df = dataset_measure_model_df.melt(
                    id_vars=['measure', 'model', 'metric']
                ).query('variable.str.contains("score")').dropna().rename(
                    columns={
                        'variable': 'Comparison Model',
                    }
                )
                plot_model_dataset_measure_heatmap_df(
                    dataset_measure_model_melt_df,
                    model,
                    dataset,
                    measure,
                    hue_order,
                    [f'{m}_comparison_score' for m in COMPARISON_MODELS],
                    METRICS,
                    model_folder_path
                )
                plot_model_dataset_measure_df(
                    dataset_measure_model_melt_df,
                    model,
                    dataset,
                    measure,
                    hue_order,
                    [f'{m}_comparison_score' for m in COMPARISON_MODELS],
                    METRICS,
                    model_folder_path
                )
                for metric in METRICS:
                    dataset_measure_model_metric_df = \
                        dataset_measure_model_df.query(f'metric=="{metric}"')
                    dataset_measure_metric_model_melt_df = dataset_measure_model_metric_df.melt(
                        id_vars=['measure', 'model', 'metric']
                    ).query('variable.str.contains("score")').dropna().rename(
                        columns={
                            'variable': 'Comparison Model',
                            'value': f'{METRICS_PRESENTABLE[metric]} score'
                        }
                    )
                    plot_model_dataset_measure_metric_df(
                        dataset_measure_metric_model_melt_df,
                        model,
                        dataset,
                        measure,
                        metric,
                        hue_order,
                        [f'{m}_comparison_score' for m in COMPARISON_MODELS],
                        model_folder_path
                    )
        for measure in MEASURES:
            print(f"\tPlotting {measure} measure")
            dataset_measure_iter_df = dataset_df.query(f'measure=="{measure}"')[
                [str(i) for i in range(args.num_of_iterations)]
            ]
            for metric in METRICS:
                model_ids = get_model_ids(dataset_df.index)
                metric_columns = get_metric_columns(metric, model_ids)
                comparison_results = {}
                for comp_model, presentable_comp_model in COMPARISON_MODELS_PRESENTABLE.items():
                    comparison_results[f'{presentable_comp_model} on test set'] = dataset_df.query(
                        f'model=="model_test" & metric=="{metric}" & measure=="{measure}"'
                    )[f'{comp_model}_comparison_score'].iloc[0]
                plot_dataset_measure(
                    dataset_measure_iter_df,
                    dataset,
                    measure,
                    metric,
                    metric_columns,
                    comparison_results,
                    args.output_path
                )


if '__main__' == __name__:
    main()
