import torch
from utils.gcn_rnn_net import GCNRNNNet
from utils.netscape_model import NETSCAPEModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNRNNModel(NETSCAPEModel):
    def __init__(self, parameters, graph_series_data):
        super(GCNRNNModel, self).__init__(parameters, graph_series_data)

    def _get_internal_net(self):
        return GCNRNNNet(
            self._graph_series_data.get_number_of_features(),
            self._params["gcn_latent_dim"],
            self._params["gcn_hidden_sizes"],
            self._params["gcn_dropout_rate"],
            self._params["lstm_hidden_size"],
            self._params["lstm_num_layers"],
            self._params["lstm_dropout_rate"],
        )

    def _evaluate_secondary_metric(
            self,
            secondary_metric,
            should_evaluate,
            val_output,
            evaluation_set_learned_labels,
            evaluation_set_current_labels,
            evaluation_set_next_labels
    ):
        metric = None
        if should_evaluate:
            metric = secondary_metric(
                evaluation_set_current_labels + val_output, evaluation_set_next_labels)

        return metric