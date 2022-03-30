import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCNNet(nn.Module):
    def __init__(self, num_features, gcn_latent_dim, h_layers=[16], dropout=0.5):
        super(GCNNet, self).__init__()
        self._convs = nn.ModuleList()
        self._convs.append(GCNConv(num_features, h_layers[0], add_self_loops=False))
        for idx, layer in enumerate(h_layers[1:]):
            self._convs.append(GCNConv(h_layers[idx], layer, add_self_loops=False))
        self._convs.append(GCNConv(h_layers[-1], gcn_latent_dim, add_self_loops=False))
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

class GCNSeriesNet(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        gcn_latent_dim: int,
        gcn_hidden_sizes: list,
        gcn_dropout_rate: float,
    ):
        super(GCNSeriesNet, self).__init__()
        self._device = DEVICE

        self._num_of_features = num_of_features
        self._gcn_latent_dim = gcn_latent_dim
        self._gcn_hidden_sizes = gcn_hidden_sizes
        self._gcn_dropout_rate = gcn_dropout_rate
        self._gcn_net = GCNNet(
            num_of_features,
            gcn_latent_dim,
            gcn_hidden_sizes,
            gcn_dropout_rate,
        ).to(self._device)

        return

    def forward(self, data: list, idx_subset: list):
        gcn_outputs = []
        for d in data:
            current_output = self._gcn_net(d)
            gcn_outputs.append(current_output)
        gcn_output = torch.stack(gcn_outputs, dim=0)
        gcn_output_only_training = gcn_output[:, idx_subset, :]

        return gcn_output_only_training

