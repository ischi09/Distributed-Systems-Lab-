#!/bin/bash

#SBATCH --output=/home/ishamanna/Distributed-Systems-Lab-/results%j.out
#SBATCH --error=/home/ishamanna/Distributed-Systems-Lab-/results%j.err

old_dir=$(pwd)
cd ../src

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.results_out=deepsets_mlp_fspool-diff-set-sizes-results.csv \
experiment.use_gpu=True \
task.label=sum,largest_pair_sum,largest_triple_sum \
task.max_set_size=8,16,32,64,128,256,512,1024 \
task.multisets=True \
model.type=deepsets_mlp_fspool

cd $old_dir
