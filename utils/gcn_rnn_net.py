import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn, optim

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
