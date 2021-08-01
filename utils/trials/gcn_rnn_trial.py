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
        self._argparser.add_argument("--graph-cutoff-number", type=int,
                                     help="Optional cutoff of number of time steps to use")

    def _update_specific_parser_arguments(self, args):
        if args.graph_cutoff_number is not None:
            self._params["graphs_cutoff_number"] = args.graph_cutoff_number
        if self._nni:
            p = nni.get_next_parameter()
            p["gcn_hidden_sizes"] = ast.literal_eval(p["gcn_hidden_sizes"])
            self._params.update(p)

    def _get_trial_name(self):
        return "GCNRNN"