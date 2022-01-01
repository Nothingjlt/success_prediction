#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` <model id> <num of iterations> \"<model command line parameters>\""
  exit 0
fi


DATASETS=('ca-cit-HepPh ca_cit_hepph' 'cambridge_haggle complab05' 'cambridge_haggle infocom05' 'cambridge_haggle infocom06_four_hours' 'cambridge_haggle infocom06_hours' 'cambridge_haggle infocom06' 'cambridge_haggle intel05' 'dnc dnc_candidate_one' 'dnc dnc_candidate_two' 'fb-wosn-friends fb-wosn-friends_1' 'fb-wosn-friends fb-wosn-friends_2' 'ia-digg-reply ia-digg-reply' 'ia-retweet-pol ia_retweet_pol' 'ia-slashdot-reply-dir ia-slashdot-reply-dir' 'reality_mining reality_mining_daily' 'reality_mining reality_mining_monthly')
ELEMENTS=${#DATASETS[@]}

mkdir ./out/$1

for (( i=0;i<$ELEMENTS;i++)); do
    IFS=' ' read -ra DATASET <<< "${DATASETS[${i}]}"
    echo "Evaluating ${DATASET[0]}/${DATASET[1]}"
    echo "Running python ./run_gcn_rnn_trial.py --data-folder-name ${DATASET[0]} --data-name ${DATASET[1]} --learned-label all_labels --num-iterations $2 $3"
    python ./run_gcn_rnn_trial.py --data-folder-name ${DATASET[0]} --data-name ${DATASET[1]} --learned-label all_labels --num-iterations $2 $3
    # echo "Moving output data to model folder"
    # echo "Running mv -f ./out/${DATASET[0]} ./out/$1/${DATASET[0]}"
    # mv -f ./out/${DATASET[0]} ./out/$1/${DATASET[0]}
    # echo ${DATASET[0]}
    # echo ${DATASET[1]}
done

echo "Generating output csv file"
echo "Running python ./utils/general/output_file_reader.py ./out/$1_summary.csv ./out --add_train_results"
python ./utils/general/output_file_reader.py ./out/$1_summary.csv ./out --add_train_results

echo "Generating plots"
echo "Running python ./utils/general/plots_from_csv_summary.py ./out/$1_summary.csv GCNRNN ./out/plots --num-of-iterations $2"
python ./utils/general/plots_from_csv_summary.py ./out/$1_summary.csv GCNRNN ./out/plots --num-of-iterations $2

echo "Done"