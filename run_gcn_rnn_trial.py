from utils.models.gcn_rnn_model import GCNRNNModel
from utils.netscape_trial import NETSCAPETrial
import ast
import nni


class GCNRNNTrial(NETSCAPETrial):
    def __init__(self, parameters, seed=None):
        super(GCNRNNTrial, self).__init__(parameters, seed)

    def _get_model(self, graph_data):
        return GCNRNNModel(self._params, graph_data)

    def _should_learn_diff(self):
        return True

    def _cut_graphs_list(self, graphs_before_cut, labels_before_cut):
        graphs_cutoff_number = self._params["graphs_cutoff_number"]
        return graphs_before_cut[-graphs_cutoff_number:], labels_before_cut[-graphs_cutoff_number:]

    def _add_specific_parser_arguments(self):
        pass

    def _update_specific_parser_arguments(self, args):
        if self._nni:
            p = nni.get_next_parameter()
            p["gcn_hidden_sizes"] = ast.literal_eval(p["gcn_hidden_sizes"])
            self._params.update(p)

    def _get_trial_name(self):
        return "GCNRNN"


def main():
    _params = {
        "print_every_num_of_epochs": 100,
        "data_folder_name": "dnc",
        "data_name": "dnc_candidate_two",
        "graphs_cutoff_number": 2,
        "l1_lambda": 0,
        "epochs": 500,
        "gcn_dropout_rate": 0.7,
        "lstm_dropout_rate": 0,
        "gcn_hidden_sizes": [10, 10, 10, 10, 10, 10, 10, 10, 10],
        "learning_rate": 0.001,
        "weight_decay": 0,
        "gcn_latent_dim": 5,
        "lstm_hidden_size": 10,
        "lstm_num_layers": 1,
        "learned_label": None,
        "number_of_iterations_per_test": 30,
    }

    trial = GCNRNNTrial(_params)

    trial.run_full_trial()


if __name__ == "__main__":
    main()
