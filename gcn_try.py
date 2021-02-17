import networkx as nx
import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from torch import nn, optim
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import r2_score
from lib.graph_measures.features_meta.features_meta import *
from lib.graph_measures.features_infra.graph_features import GraphFeatures
from sklearn.linear_model import LinearRegression

import argparse

import ast
import nni

DEFAULT_FEATURES_META = {
    # "betweenness_centrality": FeatureMeta(
    #     BetweennessCentralityCalculator, {"betweenness"}
    # ),
    # "kcore": FeatureMeta(KCoreCalculator, {"kcore"}),
    # "load": FeatureMeta(LoadCentralityCalculator, {"load"}),
    # "pagerank": FeatureMeta(PageRankCalculator, {"page"}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
}

DEFAULT_LABEL_TO_LEARN = "general"

DEFAULT_OUT_DIR = "out"


NNI = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNNet(nn.Module):
    def __init__(self, num_features, gcn_latent_dim, h_layers=[16], dropout=0.5):
        super(GCNNet, self).__init__()
        self._convs = nn.ModuleList()
        self._convs.append(GCNConv(num_features, h_layers[0]))
        for idx, layer in enumerate(h_layers[1:]):
            self._convs.append(GCNConv(h_layers[idx], layer))
        self._convs.append(GCNConv(h_layers[-1], gcn_latent_dim))
        self._dropout = dropout
        self._activation_func = F.leaky_relu
        self._device = DEVICE

    def forward(self, data):
        x, adj_mx = data.x.to(self._device), data.edge_index.to(self._device)
        for conv in self._convs[:-1]:
            x = conv(x, adj_mx)
            x = self._activation_func(x)
            x = F.dropout(x, p=self._dropout)
        x = self._convs[-1](x, adj_mx)
        return x


class GCNRNNNet(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        gcn_latent_dim: int,
        gcn_hidden_sizes: list,
        gcn_dropout_rate: float,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout_rate: float,
    ):
        super(GCNRNNNet, self).__init__()
        self._device = DEVICE

        self._num_of_features = num_of_features
        self._gcn_latent_dim = gcn_latent_dim
        self._gcn_hidden_sizes = gcn_hidden_sizes
        self._gcn_dropout_rate = gcn_dropout_rate
        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_dropout_rate = lstm_dropout_rate
        gcn_net = GCNNet(
            num_of_features,
            gcn_latent_dim,
            gcn_hidden_sizes,
            gcn_dropout_rate,
        ).to(self._device)
        lstm_net = nn.LSTM(
            gcn_latent_dim,
            lstm_hidden_size,
            lstm_num_layers,
            dropout=lstm_dropout_rate,
        ).to(self._device)
        head_net = nn.Linear(2 * lstm_num_layers * lstm_hidden_size, 1)

        self._modules_dict = nn.ModuleDict(
            {"gcn_net": gcn_net, "lstm_net": lstm_net, "head_net": head_net}
        ).to(self._device)

        return

    def forward(self, data: list, idx_subset: list):
        gcn_output = torch.empty(size=(1, data[0].num_nodes, self._gcn_latent_dim)).to(
            self._device
        )
        for d in data:
            gcn_output = torch.cat(
                (gcn_output, self._modules_dict["gcn_net"](
                    d)[None, :, :]), dim=0
            )
        gcn_output = gcn_output[1:, :, :]
        gcn_output_only_training = gcn_output[:, idx_subset, :]

        lstm_output, (lstm_hn, lstm_cn) = self._modules_dict["lstm_net"](
            gcn_output_only_training
        )

        lstm_hn_shape = lstm_hn.shape
        lstm_cn_shape = lstm_cn.shape

        head_output = self._modules_dict["head_net"](
            torch.cat(
                (
                    lstm_hn.reshape(
                        (lstm_hn_shape[1], lstm_hn_shape[0] * lstm_hn_shape[2])
                    ),
                    lstm_cn.reshape(
                        (lstm_cn_shape[1], lstm_cn_shape[0] * lstm_cn_shape[2])
                    ),
                ),
                dim=1,
            )
        )

        return head_output.reshape(-1)


class Model:
    def __init__(self, parameters):
        self._params = parameters
        self._device = DEVICE
        self._gcn_data_list = None
        self._criterion = nn.MSELoss()
        # self._accuracy_metric = nn.L1Loss()
        self.train_idx = None
        self.test_idx = None
        self.validation_idx = None
        self._gcn_rnn_net = None
        self._optimizer = None

    def load_data(
        self,
        gnxs: list,
        feature_matrix,
        current_timestamp_labels=None,
        next_timestamp_labels=None,
        learned_label: str = "",
        train=None,
        test=None,
        validation=None,
    ):
        self._gcn_data_list = []

        self._learned_label = learned_label

        all_nodes_set = set()
        for gnx in gnxs:
            all_nodes_set.update(gnx.nodes())
        all_nodes_list = sorted(all_nodes_set)

        self._num_of_nodes = len(all_nodes_list)

        self._node_id_to_idx = {x: i for i, x in enumerate(all_nodes_list)}
        self._idx_to_node_id = {i: x for i, x in enumerate(all_nodes_list)}

        for gnx in gnxs:
            gnx.add_nodes_from(all_nodes_set)
            nodes = gnx.nodes()
            x = torch.tensor(
                np.vstack([feature_matrix[self._node_id_to_idx[node]]
                           for node in nodes]),
                device=self._device,
            )

            edges = torch.tensor(
                np.vstack(
                    [
                        [self._node_id_to_idx[x[0]] for x in gnx.edges],
                        [self._node_id_to_idx[x[1]] for x in gnx.edges],
                    ]
                ),
                dtype=torch.long,  # Required by torch_geometric.data.Data
                device=self._device,
            )

            d = Data(x=x, edge_index=edges)

            self._gcn_data_list.append(d)

        self.train_idx = [self._node_id_to_idx[node] for node in train]
        self.test_idx = [self._node_id_to_idx[node] for node in test]
        self.validation_idx = [self._node_id_to_idx[node]
                               for node in validation]

        self.train_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self.train_idx)
        self.train_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self.train_idx)

        self.test_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self.test_idx)
        self.test_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self.test_idx)

        self.validation_next_time_labels = self._get_labels_by_indices(
            next_timestamp_labels, self.validation_idx)
        self.validation_current_labels = self._get_labels_by_indices(
            current_timestamp_labels, self.validation_idx)

        # Special case where both in_deg and out_deg are learned, reduce to deg.
        if self._learned_label == "general":

            # Trials with Yoram
            self.train_labels_to_learn = self.train_next_time_labels - self.train_current_labels
            self.test_labels_to_learn = self.test_next_time_labels - self.test_current_labels
            self.validation_labels_to_learn = self.validation_next_time_labels - \
                self.validation_current_labels

            # End trials with Yoram
            # self.train_next_time_labels = self.train_next_time_labels.sum(
            #     dim=1).log()
            # self.test_next_time_labels = self.test_next_time_labels.sum(
            #     dim=1).log()
            # self.validation_next_time_labels = self.validation_next_time_labels.sum(
            #     dim=1).log()

        self.train_idx = [self._node_id_to_idx[node] for node in train]
        self.test_idx = [self._node_id_to_idx[node] for node in test]
        self.validation_idx = [self._node_id_to_idx[node]
                               for node in validation]

        self._gcn_rnn_net = GCNRNNNet(
            feature_matrix.shape[1],
            self._params["gcn_latent_dim"],
            self._params["gcn_hidden_sizes"],
            self._params["gcn_dropout_rate"],
            self._params["lstm_hidden_size"],
            self._params["lstm_num_layers"],
            self._params["lstm_dropout_rate"],
        )

        self._optimizer = optim.Adam(
            self._gcn_rnn_net.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=self._params["weight_decay"],
        )

        self._l1_lambda = self._params["l1_lambda"]

        print(
            f"number of params in model: {sum(p.numel() for p in self._gcn_rnn_net.parameters())}")

        return

    def _get_labels_by_indices(self, labels, indices):
        ret_labels = torch.tensor(
            [labels[self._learned_label].features[self._idx_to_node_id[i]]
                for i in indices],
            dtype=torch.float,
            device=self._device,
        )
        # Special case where rank is required
        if self._learned_label == "general":
            ret_labels = ret_labels.sum(dim=1)
        return ret_labels

    @property
    def data(self):
        return self._gcn_data_list.clone()

    def _ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def _l1_norm(self):
        l1_norm = 0
        for p in self._gcn_rnn_net.parameters():
            l1_norm += p.abs().sum()
        return l1_norm

    def evaluate(self, evaluation_set: str = "test"):
        # self._gcn_rnn_net.eval()

        if evaluation_set == "test":
            self._gcn_rnn_net.eval()
            evaluation_set_idx = self.test_idx
            # Trials with Yoram
            # evaluation_set_labels = self.test_next_time_labels
            evaluation_set_learned_labels = self.test_labels_to_learn
            evaluation_set_current_labels = self.test_current_labels
            evaluation_set_next_labels = self.test_next_time_labels
            # End trial with Yoram
        elif evaluation_set == "validation":
            self._gcn_rnn_net.eval()
            evaluation_set_idx = self.validation_idx
            # Trials with Yoram
            # evaluation_set_labels = self.validation_next_time_labels
            evaluation_set_learned_labels = self.validation_labels_to_learn
            evaluation_set_current_labels = self.validation_current_labels
            evaluation_set_next_labels = self.validation_next_time_labels
            # End trial with Yoram
        elif evaluation_set == "train":
            evaluation_set_idx = self.train_idx
            # Trials with Yoram
            # evaluation_set_labels = self.train_next_time_labels
            evaluation_set_learned_labels = self.train_labels_to_learn
            evaluation_set_current_labels = self.train_current_labels
            evaluation_set_next_labels = self.train_next_time_labels
            # End trial with Yoram
        else:
            print(f"Invalid evaluation set: {evaluation_set}")
            return

        val_output = self._gcn_rnn_net(self._gcn_data_list, evaluation_set_idx)
        val_loss = self._criterion(val_output, evaluation_set_learned_labels)

        val_loss += self._l1_lambda * self._l1_norm()

        accuracy = r2_score(
            val_output.cpu().detach().numpy(),
            evaluation_set_learned_labels.float().cpu().detach().numpy(),
        )

        tot_accuracy = r2_score(
            (evaluation_set_current_labels + val_output).cpu().detach().numpy(),
            evaluation_set_next_labels.float().cpu().detach().numpy()
        )

        return val_loss, accuracy, tot_accuracy

    def train(self):
        (
            train_loss_list,
            train_accuracy_list,
            train_tot_accuracy_list,
            val_accuracy_list,
            val_loss_list,
            val_tot_accuracy_list
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for epoch in range(int(self._params["epochs"])):
            self._gcn_rnn_net.train()
            self._optimizer.zero_grad()

            # output = self._gcn_rnn_net(self._gcn_data_list, self.train_idx)
            # loss = self._criterion(output, self.train_labels)
            # train_accuracy = r2_score(output.cpu().detach().numpy(), self.train_labels.float().cpu().detach().numpy())
            train_loss, train_accuracy, train_tot_accuracy = self.evaluate("train")
            train_loss_list.append(train_loss.data.cpu().item())
            train_accuracy_list.append(train_accuracy)
            train_tot_accuracy_list.append(train_tot_accuracy)
            train_loss.backward()
            self._optimizer.step()

            # Evaluate validation set
            self._gcn_rnn_net.eval()
            validation_loss, validation_accuracy, validation_tot_accuracy = self.evaluate(
                "validation")
            val_loss_list.append(validation_loss.data.cpu().item())
            val_accuracy_list.append(validation_accuracy)
            val_tot_accuracy_list.append(validation_tot_accuracy)

            if epoch % 100 == 99:
                print(
                    f"epoch: {epoch + 1}, train loss: {train_loss_list[-1]:.5f}, validation loss:{val_loss_list[-1]:.5f}, "
                    f"train accuracy: {train_accuracy_list[-1]:.5f}, validation accuracy: {val_accuracy_list[-1]:.5f} "
                    f"train tot accuracy: {train_tot_accuracy_list[-1]:.5f}, validation tot accuracy: {val_tot_accuracy_list[-1]:.5f} "
                )
        self._gcn_rnn_net.eval()
        train_loss, train_accuracy, train_tot_accuracy = self.evaluate(
            "train")
        if NNI:
            nni.report_intermediate_result(train_accuracy)
        return

    def evaluate_zero_model_total_num(self, labels):
        all_indices = set()
        all_indices = all_indices.union(self.train_idx).union(
            self.test_idx).union(self.validation_idx)
        labels = torch.stack(
            list(map(lambda x: self._get_labels_by_indices(x, all_indices), labels)))
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        for idx, l in enumerate(labels[:-1]):
            true_labels.append(labels[idx + 1])
            predictions.append(l)

            losses.append(self._criterion(predictions[-1], true_labels[-1]))
            accuracies.append(r2_score(
                predictions[-1].cpu().detach().numpy(), true_labels[-1].cpu().detach().numpy()))
        return losses, accuracies

    def evaluate_zero_model_diff(self, labels):
        all_indices = set()
        all_indices = all_indices.union(self.train_idx).union(
            self.test_idx).union(self.validation_idx)
        labels = torch.stack(
            list(map(lambda x: self._get_labels_by_indices(x, all_indices), labels)))
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        for idx, l in enumerate(labels[:-1]):
            true_labels.append(labels[idx + 1]-l)
            predictions.append(torch.zeros(len(l), device=DEVICE))

            losses.append(self._criterion(predictions[-1], true_labels[-1]))
            accuracies.append(r2_score(
                predictions[-1].cpu().detach().numpy(), true_labels[-1].cpu().detach().numpy()))
        return losses, accuracies

    def evaluate_first_order_model_total_number(self, labels):
        all_indices = set()
        all_indices = all_indices.union(self.train_idx).union(
            self.test_idx).union(self.validation_idx)
        labels = torch.stack(
            list(map(lambda x: self._get_labels_by_indices(x, all_indices), labels)))
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        lin_reg_models = []
        time_steps = np.arange(len(labels)-1).reshape(-1, 1)
        for idx, n in enumerate(labels.T):
            lin_reg_models.append(
                LinearRegression().fit(time_steps, n[:-1].cpu()))

        for idx, l in enumerate(labels[:-1]):
            true_labels.append(labels[idx + 1])
            predictions.append(torch.tensor([m.predict(
                np.array([[idx+1]])) for m in lin_reg_models], device=DEVICE).view(-1))

            losses.append(self._criterion(predictions[idx], true_labels[idx]))
            accuracies.append(
                r2_score(predictions[idx].cpu().detach().numpy(), true_labels[idx].cpu().detach().numpy()))

        return losses, accuracies

    def evaluate_first_order_model_diff(self, labels):
        all_indices = set()
        all_indices = all_indices.union(self.train_idx).union(
            self.test_idx).union(self.validation_idx)
        labels = torch.stack(
            list(map(lambda x: self._get_labels_by_indices(x, all_indices), labels)))
        losses = []
        predictions = []
        true_labels = []
        accuracies = []
        lin_reg_models = []
        time_steps = np.arange(len(labels)-1).reshape(-1, 1)
        for idx, n in enumerate(labels.T):
            diffs = []
            for idx, m in enumerate(n[:-1]):
                diffs.append(n[idx+1]-m)
            lin_reg_models.append(
                LinearRegression().fit(time_steps, diffs))

        for idx, l in enumerate(labels[:-1]):
            true_labels.append(labels[idx + 1]-l)
            predictions.append(torch.tensor([m.predict(
                np.array([[idx+1]])) for m in lin_reg_models], device=DEVICE).view(-1))

            losses.append(self._criterion(predictions[idx], true_labels[idx]))
            accuracies.append(
                r2_score(predictions[idx].cpu().detach().numpy(), true_labels[idx].cpu().detach().numpy()))

        return losses, accuracies


def train_test_split(graphs):
    prev_graph = graphs[0]
    all_intersection = np.array(prev_graph.nodes())
    for g in graphs[1:]:
        all_intersection = np.intersect1d(all_intersection, g.nodes())
        prev_graph = g

    print("# of nodes that appear at all timestamps", len(all_intersection))
    val_test_inds = np.random.choice(
        all_intersection, round(len(all_intersection) * 0.2), replace=False
    )
    test, validation = np.array_split(val_test_inds, 2)
    train = np.setdiff1d(np.array(all_intersection), val_test_inds)
    e = 0
    return train, test, validation


def get_measures_from_graphs(
    graphs: list,
    features_meta: dict = DEFAULT_FEATURES_META,
    dir_path: str = DEFAULT_OUT_DIR,
) -> list:
    labels = []
    for g in graphs:
        features = GraphFeatures(g, features_meta, dir_path)
        features.build()
        labels.append(features)
    return labels


def load_input(parameters: dict):
    graphs = pickle.load(
        open(
            "./pickles/" + str(parameters["data_name"]) +
            "/dnc_candidate_two.pkl", "rb"
        )
    )
    # labels = pickle.load(open("./pickles/" + str(parameters["data_name"]) + "/dnc_with_labels_candidate_one.pkl", "rb"))
    labels = get_measures_from_graphs(graphs)
    all_nodes = set()
    for g in graphs:
        all_nodes.update(g.nodes())
    feature_mx = torch.eye(len(all_nodes))
    adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]
    return graphs, labels, feature_mx, adjacency_matrices


def run_trial(parameters):
    print(parameters)
    learned_label = parameters["learned_label"]
    graphs, labels, feature_mx, adjacency_matrices = load_input(parameters)
    train, test, validation = train_test_split(graphs)

    model_loss = []
    model_accuracy = []
    model_tot_accuracy = []

    model = Model(parameters)
    model.load_data(
        graphs[:1],
        feature_mx,
        labels[0],
        labels[1],
        learned_label,
        train,
        test,
        validation,
    )

    for idx, g in enumerate(graphs[:-1]):
        model = Model(parameters)
        model.load_data(
            [g],
            feature_mx,
            labels[idx],
            labels[idx + 1],
            learned_label,
            train,
            test,
            validation,
        )
        model.train()

        test_loss, test_accuracy, test_tot_accuracy = model.evaluate()

        model_loss.append(test_loss)
        model_accuracy.append(test_accuracy)
        model_tot_accuracy.append(test_tot_accuracy)

    zero_model_tot_loss, zero_model_tot_accuracy = model.evaluate_zero_model_total_num(
        labels)
    first_order_tot_loss, first_order_tot_accuracy = model.evaluate_first_order_model_total_number(
        labels)
    zero_model_diff_loss, zero_model_diff_accuracy = model.evaluate_zero_model_diff(
        labels)
    first_order_diff_loss, first_order_diff_accuracy = model.evaluate_first_order_model_diff(
        labels)

    print(
        f"test loss: {np.mean([m.data.item() for m in model_loss]):.5f}, test_tot_accuracy: {np.mean(model_tot_accuracy):.5f}, test_diff_accuracy: {np.mean(model_accuracy):.5f}")
    print(
        f"zero model loss: {np.mean([m.data.item() for m in zero_model_tot_loss]):.5f}, zero model accuracy: {np.mean(zero_model_tot_accuracy):.5f}")
    print(
        f"first order model loss: {np.mean([m.data.item() for m in first_order_tot_loss]):.5f}, first order model accuracy: {np.mean(first_order_tot_accuracy):.5f}")
    print(
        f"zero diff model loss: {np.mean([m.data.item() for m in zero_model_diff_loss]):.5f}, zero diff model accuracy: {np.mean(zero_model_diff_accuracy):.5f}")
    print(
        f"first order diff model loss: {np.mean([m.data.item() for m in first_order_diff_loss]):.5f}, first order diff model accuracy: {np.mean(first_order_diff_accuracy):.5f}")
    print("\n")

    return (
        [m.data.item() for m in model_loss],
        model_accuracy,
        model_tot_accuracy,
        [m.data.item() for m in zero_model_tot_loss],
        zero_model_tot_accuracy,
        [m.data.item() for m in first_order_tot_loss],
        first_order_tot_accuracy,
        [m.data.item() for m in zero_model_diff_loss],
        zero_model_diff_accuracy,
        [m.data.item() for m in first_order_diff_loss],
        first_order_diff_accuracy
    )


if __name__ == "__main__":
    _params = {
        "data_name": "dnc",
        "net": GCNNet,
        "l1_lambda": 0,
        "epochs": 1000,
        "gcn_dropout_rate": 0.5,
        "lstm_dropout_rate": 0,
        "gcn_hidden_sizes": [40, 40],
        "learning_rate": 0.0005,
        "weight_decay": 0.00003,
        "gcn_latent_dim": 10,
        "lstm_hidden_size": 10,
        "lstm_num_layers": 1,
        "learned_label": DEFAULT_LABEL_TO_LEARN,
    }
    l1_lambda = [0, 1e-7]
    epochs = [500]
    gcn_dropout_rate = [0.3, 0.5]
    gcn_hidden_sizes = [[100, 100], [200, 200]]
    learning_rate = [1e-3, 1e-2, 3e-2]
    weight_decay = [5e-2, 1e-2]
    gcn_latent_dim = [50, 100]
    lstm_hidden_size = [50, 100]
    results = []

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--nni", action='store_true')

    args = argparser.parse_args()

    NNI = args.nni

    if NNI:
        p = nni.get_next_parameter()
        p["gcn_hidden_sizes"] = ast.literal_eval(p["gcn_hidden_sizes"])
        _params.update(p)

    (
        model_loss,
        model_accuracy,
        model_tot_accuracy,
        zero_model_tot_loss,
        zero_model_tot_accuracy,
        first_order_tot_loss,
        first_order_tot_accuracy,
        zero_model_diff_loss,
        zero_model_diff_accuracy,
        first_order_diff_loss,
        first_order_diff_accuracy
    ) = run_trial(_params)

    if NNI:
        nni.report_final_result(np.mean(model_tot_accuracy))

