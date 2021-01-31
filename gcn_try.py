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


class GCNNet(nn.Module):
    def __init__(self, num_features, num_classes, h_layers=[16], dropout=0.5):
        super(GCNNet, self).__init__()
        self._conv1 = GCNConv(num_features, h_layers[0])
        self._conv2 = GCNConv(h_layers[0], num_classes)
        self._dropout = dropout
        self._activation_func = F.relu
        # self._device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, Data):
        x, adj_mx = Data.x.to(self._device), Data.edge_index.to(self._device)
        x = self._conv1(x, adj_mx)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._dropout)
        x = self._conv2(x, adj_mx)
        # x = F.softmax(x, dim=1)
        return x


class Model:
    def __init__(self, parameters):
        self._params = parameters
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._data = None
        self._criterion = self._ce_loss
        self.activation = nn.ReLU

    def load_data(self, gnx, labels, feature_matrix, train, test, validation):
        nodes = sorted(gnx.nodes)
        dict = {x: i for i, x in enumerate(nodes)}
        x = torch.tensor(np.vstack([feature_matrix[dict[node]] for node in nodes]), device=self._device)
        edges = torch.tensor(np.vstack([[dict[x[0]] for x in gnx.edges],
                                        [dict[x[1]] for x in gnx.edges]]), dtype=torch.long, device=self._device)

        self._data = Data(x=x, edge_index=edges)
        self._data.train_idx = [dict[node] for node in train]
        self._data.test_idx = [dict[node] for node in test]
        self._data.validation_idx = [dict[node] for node in validation]

        self._data.train_labels = torch.tensor([labels['degree'][i] for i in self._data.train_idx],dtype=torch.double, device=self._device)
        self._data.test_labels = torch.tensor([labels['degree'][i] for i in self._data.test_idx],dtype=torch.double, device=self._device)
        self._data.validation_labels = torch.tensor([labels['degree'][i] for i in self._data.validation_idx],dtype=torch.double, device=self._device)

        self._num_features = x.shape[1]
        self._num_classes = self._params['num_of_classes']
        self._net = self._params['net'](self._num_features,
                                        self._num_classes,
                                        h_layers=self._params['hidden_sizes'],
                                        dropout=self._params['dropout_rate'])
        self._net.to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(),
                                     lr=self._params['learning_rate'],
                                     weight_decay=self._params['weight_decay'])
        self.out = []
        return self._data

    @property
    def data(self):
        return self._data.clone()

    def _ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def train(self):
        self._net.train()

        train_loss, val_loss, train_acc, val_acc = [], [], [], []

        for epoch in range(int(self._params["epochs"])):
            self._optimizer.zero_grad()
            output = self._net(self._data)
            output = output[self._data.train_idx, :]
            loss = self._criterion(output,self._data.y[self._data.train_idx])
            mse_for_acc = torch.nn.MSELoss()
            mse_for_acc = mse_for_acc(output, self._data.y[self._data.train_idx].float())
            train_loss.append(loss.data.cpu().item())
            train_acc.append(mse_for_acc.data.cpu().item())
            loss.backward()
            self._optimizer.step()

            # valid
            self._net.eval()
            val_output = self._net(self._data)
            val_output = val_output[self._data.validation_idx, :]
            val_loss = self._criterion(val_output, self._data.y[self._data.validation_idx])
            mse_for_acc_val = torch.nn.MSELoss()
            mse_for_acc_val = mse_for_acc_val(val_output, self._data.y[self._data.validation_idx].float())
            val_loss.append(val_loss.data.cpu().item())
            val_acc.append(mse_for_acc_val.data.cpu().item())

            print("epoch: {}, train loss: {:.5f}, test loss:{:.5f}, train mse acc: {:.5f}, test mse acc: {:.5f} ".format(
                    epoch + 1, loss, val_loss, mse_for_acc, mse_for_acc_val))
            self._net.train()
        return output

def train_test_split():
    #indices that appear in all timestamps.
    indices = [1, 5, 12, 14, 15, 21, 24, 25, 31, 32, 34, 45, 47, 56, 80, 87, 90,
               99, 104, 120, 129, 137, 151, 153, 155, 165, 177, 195, 203, 205, 210,
               217, 220, 223, 228, 245, 246, 252, 263, 266, 272, 278, 279, 293, 306, 312,
               329, 341, 345, 346, 350, 358, 362, 363, 385, 403, 410, 411, 412, 419, 429, 434, 449, 453,
               457, 465, 467, 479, 498, 511, 521, 536, 539, 547, 549, 553, 556, 558, 572, 574, 585, 599,
               601, 605, 620, 628, 630, 634, 641, 689, 690, 691, 702, 705, 713, 718, 723, 730, 737, 740,
               764, 793, 794, 795, 803, 805, 817, 827, 829, 833, 838, 841, 852, 853, 863, 869, 882, 890,
               895, 897, 898, 903, 904, 907, 939, 943, 956, 980, 987, 991, 993, 998, 999, 1019, 1037, 1052,
               1058, 1070, 1073, 1080, 1095, 1116, 1120, 1135, 1144, 1151, 1157, 1158, 1159, 1164, 1179, 1181,
               1192, 1194, 1195, 1198, 1214, 1222, 1241, 1246, 1258, 1269, 1274, 1277, 1278, 1287, 1291, 1300,
               1302, 1306, 1309, 1323, 1328, 1369, 1374, 1377, 1380, 1384, 1385, 1393, 1430, 1434, 1440, 1441,
               1444, 1451, 1453, 1465, 1467, 1480, 1500, 1501, 1505, 1510, 1522, 1550, 1576, 1578, 1580, 1586,
               1587, 1594, 1601, 1618, 1619, 1628, 1629, 1641, 1663, 1669, 1670, 1690, 1694, 1701, 1706, 1720,
               1728, 1731, 1743, 1751, 1758, 1767, 1768, 1770, 1780, 1792, 1797, 1799, 1800, 1809, 1813, 1814,
               1830, 1839, 1844, 1854, 1874, 1876, 1878, 1881, 1882, 1889, 1894, 1899, 1906, 1911, 1912, 1930,
               1932, 1952, 1955, 1956, 1957, 1963, 1974, 1976, 1981, 1983, 1987, 1992, 1995, 1998, 1999, 2006,
               2010, 2027]

    print(len(indices))
    val_test_inds = np.random.choice(indices, round(len(indices) * 0.8), replace=False)
    test, validation = np.array_split(val_test_inds, 2)
    train = np.setdiff1d(np.array(indices), val_test_inds)
    return train, test, validation


def load_input(parameters):
    graphs = pickle.load(open("./pickles/" + str(parameters["data_name"]) + "/dnc_candidate_one.pkl", "rb"))
    # labels = pickle.load(open("./pickles/" + str(parameters["data_name"]) + "/labels.pkl", "rb"))
    feature_mx = [torch.eye(g.number_of_nodes()) for g in graphs]
    adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]
    return graphs, labels, feature_mx, adjacency_matrices


def run_trial(parameters):
    print(parameters)
    graphs, labels, feature_mx, adjacency_matrices = load_input(parameters)
    train, test, validation = train_test_split()

    out = []
    out_train = []
    total_out = []
    for i in range(len(graphs)):
        model = Model(parameters)
        model.load_data(graphs[i],labels[i],feature_mx[i], train, test, validation)
        out_train.append(model.train())
        all_out, out_test = model.test()
        out.append(out_test)
        total_out.append(all_out)


if __name__ == '__main__':
    params_ = {"data_name": "dnc",
               "net": GCNNet,
               "epochs": 30,
               "activation": "relu",
               "dropout_rate": 0.3,
               "hidden_sizes": [10],
               "learning_rate": 0.03,
               "weight_decay": 0.005,
               "num_of_classes": 1
               }
    run_trial(params_)
