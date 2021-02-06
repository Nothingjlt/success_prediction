import networkx as nx
import torch
import os
import pickle
import numpy as np
from torch_geometric.data import Data
from torch import nn, optim
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from lib.graph_measures.features_meta.features_meta import *
from lib.graph_measures.features_infra.graph_features import GraphFeatures

DEFAULT_FEATURES_META = {
    # "betweenness_centrality": FeatureMeta(
    #     BetweennessCentralityCalculator, {"betweenness"}
    # ),
    "kcore": FeatureMeta(KCoreCalculator, {"kcore"}),
    # "load": FeatureMeta(LoadCentralityCalculator, {"load"}),
    # "pagerank": FeatureMeta(PageRankCalculator, {"page"}),
    # "general": FeatureMeta(GeneralCalculator, {"gen"}),
}

DEFAULT_LABEL_TO_LEARN = "kcore"

DEFAULT_OUT_DIR = "out"


class GCNNet(nn.Module):
    def __init__(self, num_features, gcn_latent_dim, h_layers=[16], dropout=0.5):
        super(GCNNet, self).__init__()
        self._conv1 = GCNConv(num_features, h_layers[0])
        self._conv2 = GCNConv(h_layers[0], gcn_latent_dim)
        self._dropout = dropout
        self._activation_func = F.leaky_relu
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        x, adj_mx = data.x.to(self._device), data.edge_index.to(self._device)
        x = self._conv1(x, adj_mx)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._dropout)
        x = self._conv2(x, adj_mx)
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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                (gcn_output, self._modules_dict["gcn_net"](d)[None, :, :]), dim=0
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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._gcn_data_list = None
        self._criterion = nn.MSELoss()
        self._accuracy_metric = nn.L1Loss()

    def load_data(
        self,
        gnxs: list,
        feature_matrix,
        labels,
        learned_label: str,
        train,
        test,
        validation,
    ):
        self._gcn_data_list = []

        all_nodes_set = set()
        for gnx in gnxs:
            all_nodes_set.update(gnx.nodes())
        all_nodes_list = sorted(all_nodes_set)

        self._num_of_nodes = len(all_nodes_list)

        node_id_to_idx = {x: i for i, x in enumerate(all_nodes_list)}
        idx_to_node_id = {i: x for i, x in enumerate(all_nodes_list)}

        for gnx in gnxs:
            gnx.add_nodes_from(all_nodes_set)
            nodes = gnx.nodes()
            x = torch.tensor(
                np.vstack([feature_matrix[node_id_to_idx[node]] for node in nodes]),
                device=self._device,
            )

            edges = torch.tensor(
                np.vstack(
                    [
                        [node_id_to_idx[x[0]] for x in gnx.edges],
                        [node_id_to_idx[x[1]] for x in gnx.edges],
                    ]
                ),
                dtype=torch.long,  # Required by torch_geometric.data.Data
                device=self._device,
            )

            d = Data(x=x, edge_index=edges)

            self._gcn_data_list.append(d)

        self.train_idx = [node_id_to_idx[node] for node in train]
        self.test_idx = [node_id_to_idx[node] for node in test]
        self.validation_idx = [node_id_to_idx[node] for node in validation]

        self.train_labels = torch.tensor(
            [labels[learned_label].features[idx_to_node_id[i]] for i in self.train_idx],
            dtype=torch.float,
            device=self._device,
        )
        self.test_labels = torch.tensor(
            [labels[learned_label].features[idx_to_node_id[i]] for i in self.test_idx],
            dtype=torch.float,
            device=self._device,
        )
        self.validation_labels = torch.tensor(
            [
                labels[learned_label].features[idx_to_node_id[i]]
                for i in self.validation_idx
            ],
            dtype=torch.float,
            device=self._device,
        )

        # Special case where both in_deg and out_deg are learned, reduce to deg.
        if learned_label == "general":
            self.train_labels = self.train_labels.sum(dim=1)
            self.test_labels = self.test_labels.sum(dim=1)
            self.validation_labels = self.validation_labels.sum(dim=1)

        self.train_idx = [node_id_to_idx[node] for node in train]
        self.test_idx = [node_id_to_idx[node] for node in test]
        self.validation_idx = [node_id_to_idx[node] for node in validation]

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
        return self._gcn_data_list

    @property
    def data(self):
        return self._gcn_data_list.clone()

    def _ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def evaluate(self, evaluation_set: str = "test"):
        self._gcn_rnn_net.eval()

        if evaluation_set == "test":
            evaluation_set_idx = self.test_idx
            evaluation_set_labels = self.test_labels
        elif evaluation_set == "validation":
            evaluation_set_idx = self.validation_idx
            evaluation_set_labels = self.validation_labels
        elif evaluation_set == "train":
            evaluation_set_idx = self.train_idx
            evaluation_set_labels = self.train_labels
        else:
            print(f"Invalid evaluation set: {evaluation_set}")
            return

        val_output = self._gcn_rnn_net(self._gcn_data_list, evaluation_set_idx)
        val_loss = self._criterion(val_output, evaluation_set_labels)
        accuracy = self._accuracy_metric(val_output, evaluation_set_labels.float())

        return val_loss, accuracy

    def train(self):
        train_loss, val_loss, train_acc, val_accuracy_list, val_loss_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for epoch in range(int(self._params["epochs"])):
            self._gcn_rnn_net.train()
            self._optimizer.zero_grad()

            output = self._gcn_rnn_net(self._gcn_data_list, self.train_idx)

            loss = self._criterion(output, self.train_labels)
            train_accuracy = self._accuracy_metric(output, self.train_labels.float())
            train_loss.append(loss.data.cpu().item())
            train_acc.append(train_accuracy.data.cpu().item())
            loss.backward()
            self._optimizer.step()

            # Evaluate validation set
            self._gcn_rnn_net.eval()
            val_loss, validation_accuracy = self.evaluate("validation")
            val_loss_list.append(val_loss.data.cpu().item())
            val_accuracy_list.append(validation_accuracy.data.cpu().item())

            print(
                f"epoch: {epoch + 1}, train loss: {train_loss[-1]:.5f}, validation loss:{val_loss_list[-1]:.5f}, "
                f"train accuracy: {train_acc[-1]:.5f}, validation accuracy: {val_accuracy_list[-1]:.5f} "
            )
        return


def train_test_split():
    # indices that appear in all timestamps.
    indices = [
        1,
        5,
        12,
        14,
        15,
        21,
        24,
        25,
        31,
        32,
        34,
        45,
        47,
        56,
        80,
        87,
        90,
        99,
        104,
        120,
        129,
        137,
        151,
        153,
        155,
        165,
        177,
        195,
        203,
        205,
        210,
        217,
        220,
        223,
        228,
        245,
        246,
        252,
        263,
        266,
        272,
        278,
        279,
        293,
        306,
        312,
        329,
        341,
        345,
        346,
        350,
        358,
        362,
        363,
        385,
        403,
        410,
        411,
        412,
        419,
        429,
        434,
        449,
        453,
        457,
        465,
        467,
        479,
        498,
        511,
        521,
        536,
        539,
        547,
        549,
        553,
        556,
        558,
        572,
        574,
        585,
        599,
        601,
        605,
        620,
        628,
        630,
        634,
        641,
        689,
        690,
        691,
        702,
        705,
        713,
        718,
        723,
        730,
        737,
        740,
        764,
        793,
        794,
        795,
        803,
        805,
        817,
        827,
        829,
        833,
        838,
        841,
        852,
        853,
        863,
        869,
        882,
        890,
        895,
        897,
        898,
        903,
        904,
        907,
        939,
        943,
        956,
        980,
        987,
        991,
        993,
        998,
        999,
        1019,
        1037,
        1052,
        1058,
        1070,
        1073,
        1080,
        1095,
        1116,
        1120,
        1135,
        1144,
        1151,
        1157,
        1158,
        1159,
        1164,
        1179,
        1181,
        1192,
        1194,
        1195,
        1198,
        1214,
        1222,
        1241,
        1246,
        1258,
        1269,
        1274,
        1277,
        1278,
        1287,
        1291,
        1300,
        1302,
        1306,
        1309,
        1323,
        1328,
        1369,
        1374,
        1377,
        1380,
        1384,
        1385,
        1393,
        1430,
        1434,
        1440,
        1441,
        1444,
        1451,
        1453,
        1465,
        1467,
        1480,
        1500,
        1501,
        1505,
        1510,
        1522,
        1550,
        1576,
        1578,
        1580,
        1586,
        1587,
        1594,
        1601,
        1618,
        1619,
        1628,
        1629,
        1641,
        1663,
        1669,
        1670,
        1690,
        1694,
        1701,
        1706,
        1720,
        1728,
        1731,
        1743,
        1751,
        1758,
        1767,
        1768,
        1770,
        1780,
        1792,
        1797,
        1799,
        1800,
        1809,
        1813,
        1814,
        1830,
        1839,
        1844,
        1854,
        1874,
        1876,
        1878,
        1881,
        1882,
        1889,
        1894,
        1899,
        1906,
        1911,
        1912,
        1930,
        1932,
        1952,
        1955,
        1956,
        1957,
        1963,
        1974,
        1976,
        1981,
        1983,
        1987,
        1992,
        1995,
        1998,
        1999,
        2006,
        2010,
        2027,
    ]  # TODO take code from Jupyter notebook to calc this on the fly.

    print(len(indices))
    val_test_inds = np.random.choice(indices, round(len(indices) * 0.2), replace=False)
    test, validation = np.array_split(val_test_inds, 2)
    train = np.setdiff1d(np.array(indices), val_test_inds)
    return train, test, validation


def get_labels_from_graphs(
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


def load_input(parameters:dict):
    graphs = pickle.load(
        open(
            "./pickles/" + str(parameters["data_name"]) + "/dnc_candidate_two.pkl", "rb"
        )
    )
    # labels = pickle.load(open("./pickles/" + str(parameters["data_name"]) + "/dnc_with_labels_candidate_one.pkl", "rb"))
    labels = get_labels_from_graphs(graphs)
    all_nodes = set()
    for g in graphs:
        all_nodes.update(g.nodes())
    feature_mx = torch.eye(len(all_nodes))
    adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]
    return graphs, labels, feature_mx, adjacency_matrices


def run_trial(parameters):
    print(parameters)
    graphs, labels, feature_mx, adjacency_matrices = load_input(parameters)
    train, test, validation = train_test_split()

    out = []
    total_out = []

    model = Model(parameters)
    model.load_data(
        graphs[:-1],
        feature_mx,
        labels[-1],
        parameters["learned_label"],
        train,
        test,
        validation,
    )
    model.train()

    all_out, out_test = model.evaluate()

    print(
        "\n"
        f"test loss: {all_out.data.item():.5f}, test_accuracy: {out_test.data.item():.5f}"
    )


if __name__ == "__main__":
    params_ = {
        "data_name": "dnc",
        "net": GCNNet,
        "epochs": 100,
        "gcn_dropout_rate": 0.3,
        "lstm_dropout_rate": 0.3,
        "gcn_hidden_sizes": [10],
        "learning_rate": 0.03,
        "weight_decay": 0.005,
        "gcn_latent_dim": 8,
        "lstm_hidden_size": 12,
        "lstm_num_layers": 1,
        "learned_label": DEFAULT_LABEL_TO_LEARN,
    }
    run_trial(params_)
