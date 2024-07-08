#!/bin/sh

INIT_SCRIPT="scripts/environment_setup.sh"

train_epochs=10
script="scripts/model-train.py"
test_dataset="datasets/test/test_10k"

datasets=(
  "datasets/train/data_5k"
  "datasets/train/data_nop_5k"
  "datasets/train/data_10k"
  "datasets/train/data_nop_10k"
  "datasets/train/data_50k"
  "datasets/train/data_nop_50k"
  "datasets/train/data_100k"
  "datasets/train/data_nop_100k"
  "datasets/train/data_250k"
  "datasets/train/data_nop_250k"
  "datasets/train/data_500k"
  "datasets/train/data_nop_500k"
  "datasets/train/data_1kk"
  "datasets/train/data_nop_1kk"
)

sh "$INIT_SCRIPT"

if [ "$#" -gt 2 ];
then
    for train_dataset in "${datasets[@]}"; do
        echo "Training $1 -> $2 model with dataset: $train_dataset"

            case "$3" in
                true | True | TRUE | 1)
                    python $script -e $train_epochs -v -p --simplified --train "$train_dataset" --test $test_dataset -s "$1" -t "$2"
                    ;;
                false | False | FALSE | 0)
                    python $script -e $train_epochs -v -p --train "$train_dataset" --test $test_dataset -s "$1" -t "$2"
                    ;;
                *)
                    echo "Invalid 3rd argument. Please use 'true' or 'false'."
                    exit 1
                    ;;
            esac
    done
else
    echo "Usage: train.sh <source> <target> <simplified>"
    echo "Example: train.sh markdown html true"
    echo "Example: train.sh markdown html false"
    exit 1
fi
