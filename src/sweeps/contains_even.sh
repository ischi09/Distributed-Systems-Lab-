#!/bin/bash

old_dir=$(pwd)
cd ..

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.results_out=contains-even-results.csv \
task.label=contains_even \
task.multisets=True,False \
model.type=deepsets_mlp_sum,deepsets_mlp_max,deepsets_mlp_fspool,mlp,sorted_mlp,pna,small_set_transformer

cd $old_dir