#!/bin/sh

set -eu -o pipefail

INIT_SCRIPT="scripts/environment_setup.sh"

train_epochs=10
script="scripts/model-train.py"
test_dataset="datasets/test/test_10k"

datasets=(
  "datasets/train/data_5k"
  "datasets/train/data_nop_5k"
)

sources=(
  "markdown"
  "sql"
  "expression"
  "mathml"
  "ascii_math"
  "latex"
)

simplified_state=(
  true
  false
)

sh "$INIT_SCRIPT"

for train_dataset in "${datasets[@]}"; do
  for src in "${sources[@]}"; do
    if [ "$src" = "markdown" ]; then
      trg="html"
    elif [ "$src" = "sql" ]; then
      trg="kql"
    elif [ "$src" = "expression" ]; then
      trg="latex"
    elif [ "$src" = "mathml" ]; then
      trg="latex"
    elif [ "$src" = "ascii_math" ]; then
      trg="mathml"
    elif [ "$src" = "latex" ]; then
      trg="ascii_math"
    fi
    for simplified in "${simplified_state[@]}"; do
      if [ "$simplified" = true ]; then
        echo "Training $src -> $trg model with dataset: $train_dataset | simplified: $simplified"
        python $script -e $train_epochs -v -p --simplified --train "$train_dataset" --test $test_dataset -s "$src" -t "$trg"
        echo "Training $trg -> $src model with dataset: $train_dataset | simplified: $simplified"
        python $script -e $train_epochs -v -p --simplified --train "$train_dataset" --test $test_dataset -t "$src" -s "$trg"
      else
        echo "Training $src -> $trg model with dataset: $train_dataset | simplified: $simplified"
        python $script -e $train_epochs -v -p --train "$train_dataset" --test $test_dataset -s "$src" -t "$trg"
        echo "Training $trg -> $src model with dataset: $train_dataset | simplified: $simplified"
        python $script -e $train_epochs -v -p --train "$train_dataset" --test $test_dataset -t "$src" -s "$trg"
      fi
    done
  done
done
