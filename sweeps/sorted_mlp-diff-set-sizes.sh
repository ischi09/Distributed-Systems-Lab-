#!/bin/bash

old_dir=$(pwd)
cd ../src

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.results_out=sorted_mlp-diff-set-sizes-results.csv \
experiment.use_gpu=True \
task.label=sum,largest_pair_sum,largest_triple_sum \
task.max_set_size=8,16,32,64,128,256,512,1024 \
task.multisets=True \
model.type=sorted_mlp

cd $old_dir