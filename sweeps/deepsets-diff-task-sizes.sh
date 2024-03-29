#!/bin/bash

#SBATCH --output=/home/ishamanna/Distributed-Systems-Lab-/results%j.out     
#SBATCH --error=/home/ishamanna/Distributed-Systems-Lab-/results%j.err

old_dir=$(pwd)
cd ../src

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.results_out=deepsets_mlp_sum-diff-task-sizes-results.csv \
experiment.use_gpu=True \
task.label=mode,longest_seq_length,largest_contiguous_sum,max,cardinality,contains_even \
task.max_set_size=8,16,32,64,128,256,512,1024 \
task.multisets=True \
model.type=deepsets_mlp_sum

cd $old_dir