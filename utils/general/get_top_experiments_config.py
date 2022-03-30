import argparse
import csv


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('input_file_name', type=str,
                           help="path to experiment result csv file")
    argparser.add_argument('output_file_name', type=str,
                           help="path to folder to recurse and parse")
    argparser.add_argument('--get_lowest_scores', action='store_true', default=False,
                           help="Decide whether to get lowest score trials. By default get top scoring trials.")
    argparser.add_argument('--num_of_trials_to_get', type=int, default=20,
                           help='Number of trials to return.')
    argparser.add_argument('--general_command_line_arguments', type=str, default='--add-labels-of-all-times',
                           help='Special command line arguments required to reproduce nni run results')
    args = argparser.parse_args()

    print(args)

    with open(args.input_file_name) as f:
        experiment_reader = csv.DictReader(f)
        selected_trials = sorted(
            experiment_reader,
            key=lambda x: float(x['reward']),
            reverse=(not args.get_lowest_scores)
        )[:args.num_of_trials_to_get]

    lines = []

    for trial in selected_trials:
        trial_args = []
        for k, v in trial.items():
            if k == 'trialJobId' or k == 'reward':
                continue
            trial_args.append(f'--{k.replace("_", "-")} {v}')
        lines.append(f'{trial["trialJobId"]} "{args.general_command_line_arguments} {" ".join(trial_args)}"')

    print('\n'.join(lines))

    with open(args.output_file_name, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
