#!/bin/bash
INPUT="./test/INPUT.txt"
TRAIN="./test/train_dev.pkl"
MODEL="./test/model.pkl"
while getopts ":i:t:m:" opt; do
case $opt in
    i)
        echo "-i was triggered, Parameter: $OPTARG" >&2
        INPUT=$OPTARG;;
    t)
        echo "-t was triggered, Parameter: $OPTARG" >&2
        TRAIN=$OPTARG;;
    m)
        echo "-m was triggered, Parameter: $OPTARG" >&2
        MODEL=$OPTARG;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1;;
esac
done


# Inputs: path to input text file (example file is ./test/INPUT.txt)
# Function: input_to_window takes an input text file with a single sentence in it, finds all potential hedge words (from dictionary.csv and multiword_dict.csv), and outputs a text file with each potential hedge word in a window of [-2,+2] (input_test_data.txt). It also outputs a mapping of each hedge word and the character index at which it appears in the string (in the file test_data_index.txt).

#eg. This is some sort of cow.

#eg. Output file 1 (input_test_data.txt):
#0  This is some sort of
#0  is some sort of cow stop

#eg. Output file 2 (test_data_index.txt)
#19, some
#24, sort of

python ./test/input_to_window.py $INPUT

# Inputs: none
# Function: data_to_cnn_input takes the file produced at the previous step (input_test_data.txt) and puts it into a pickled format (input_test.pkl).

python ./test/data_to_cnn_input.py

# Inputs: path to the saved training data and to the model file (example files are ./test/train_dev.pkl and ./test/model.pkl)
# Function: cnn_test.py runs the CNN model in model.pkl on the input_test.pkl data. For a description of the flags and what they mean, refer to the original CNN code from Yoon Kim (github.com/yoonkim). The code outputs a input_results.pred file which just contains the output labels (1 or 0; hedge or not a hedge) for our data.

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python ./test/cnn_test.py $TRAIN $MODEL

# Inputs: path to input text file (example is ./test/INPUT.txt)
# Function: pred_to_label reads in the input_results.pred file, maps it to the index of each hedge word (from test_data_index.txt) and prints the sentence with the hedges labeled as _1 or _0 to the console.
#eg. This is some_0 sort of_0 cow.

python ./test/pred_to_label.py $INPUT