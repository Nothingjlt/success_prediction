from utils.models.gcn_rnn_model import GCNRNNModel
from utils.trials.netscape_trial import NETSCAPETrial
import ast
import nni


class GCNRNNTrial(NETSCAPETrial):
    def __init__(self, parameters, seed=None):
        super(GCNRNNTrial, self).__init__(parameters, seed)

    def _get_model(self, graph_data):
        return GCNRNNModel(self._params, graph_data)

    def _should_learn_diff(self):
        return True

    def _should_learn_logs(self):
        return True

    def _cut_graphs_list(self, graphs_before_cut, labels_before_cut, features_before_cut):
        graphs_cutoff_number = self._params["graphs_cutoff_number"]
        return graphs_before_cut[-graphs_cutoff_number:], labels_before_cut[-graphs_cutoff_number:], features_before_cut[-graphs_cutoff_number:]

    def _add_specific_parser_arguments(self):
        self._argparser.add_argument("--graphs-cutoff-number", type=int, default=2,
                                     help="Number of time steps to take into consideration, used from last to first")
        self._argparser.add_argument("--gcn-dropout-rate", type=float, default=0.7,
                                     help="Dropout rate for GCN part of the network")
        self._argparser.add_argument("--lstm-dropout-rate", type=float, default=0,
                                     help="Dropout rate for LSTM part of the network")
        self._argparser.add_argument("--gcn-hidden-sizes", type=str, default="[10, 10, 10, 10, 10, 10, 10, 10, 10]",
                                     help="Convolution sizes of GCN layers, evaluated as python-style list")
        self._argparser.add_argument("--gcn-latent-dim", type=int, default=5,
                                     help="Output dimension of GCN part of the network")
        self._argparser.add_argument("--lstm-hidden-size", type=int, default=10,
                                     help="Output dimension of LSTM part of the network")
        self._argparser.add_argument("--lstm-num-layers", type=int, default=1,
                                     help="Number of layers in LSTM part of the network")

    def _update_specific_parser_arguments(self, args):
        self._params.update(vars(args))
        if self._nni:
            p = nni.get_next_parameter()
            self._params.update(p)
        self._params["gcn_hidden_sizes"] = ast.literal_eval(self._params["gcn_hidden_sizes"])

    def _get_trial_name(self):
        return "GCNRNN"
