# %%
from math import log
import os
import pickle
import numpy as np
import seaborn as sns


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

# %%


def load_labels(data_folder_name, data_name):
    graphs, labels = pickle.load(
        open(
            "./Pickles/" + data_folder_name + "/" +
            data_name + "_with_labels" + ".pkl", "rb"
        )
    )
    all_nodes_set = set()
    for g in graphs:
        all_nodes_set.update(g.nodes())
    all_nodes_list = sorted(all_nodes_set)

    num_of_time_steps = len(labels)
    num_of_label_types = len(labels[-1].keys())
    num_of_nodes = len(all_nodes_list)

    node_id_to_idx = {x: i for i, x in enumerate(all_nodes_list)}
    idx_to_node_id = {i: x for i, x in enumerate(all_nodes_list)}

    label_types = list(labels[-1].keys())

    nodes_labels_arr = np.empty(
        (num_of_label_types, num_of_nodes, num_of_time_steps))

    for timestep, labels_timestep in enumerate(labels):
        for label_type, actual_labels in labels_timestep.items():
            for node in all_nodes_list:
                current_label = actual_labels.get(node, np.nan)
                if np.all(~np.isnan(current_label)) and type(current_label) == tuple:
                    current_label = np.sum(current_label)
                nodes_labels_arr[label_types.index(
                    label_type), node_id_to_idx[node], timestep] = current_label

    return nodes_labels_arr, label_types, all_nodes_list, node_id_to_idx, idx_to_node_id

# %%


def plot_data_evolution(data):
    p = sns.relplot(data, kind="line")


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
        p = sns.displot(x=data_1, y=data_2, rug=True, cbar=True, bins=(100,100))
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


def analyze_dataset(data_folder_name, data_name):
    nodes_labels, label_types, all_nodes_list, node_id_to_idx, idx_to_node_id = load_labels(
        data_folder_name, data_name)
    out_file_name = os.path.join("out", data_folder_name, data_name)
    generate_plots(nodes_labels, idx_to_node_id, label_types, out_file_name)


def main():
    for folder, datasets in DATASETS.items():
        for dataset in datasets:
            analyze_dataset(folder, dataset)
    return


if '__main__' == __name__:
    main()

# %%
