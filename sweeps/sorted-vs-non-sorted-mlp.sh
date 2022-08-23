#!/bin/bash

old_dir=$(pwd)
cd ../

python main.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.results_out=sorted-vs-non-sorted-mlp-results.csv \
task.max_set_size=8,16,32,64,128,256,512,1024 \
task.label=largest_triple_sum \
model.type=mlp,sorted_mlp

cd $old_dirs