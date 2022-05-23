#!/bin/bash


#SBATCH --output=/tmp/results%j.out     
#SBATCH --error=/tmp/results%j.err

old_dir=$(pwd)
cd ../src

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
paths.log=/tmp/log \
paths.checkpoints=/tmp/checkpoints \
paths.results=/tmp/results \
experiment.results_out=mlp-diff-task-sizes-classification.csv \
experiment.use_gpu=True \
task.label=mode,contains_even \
task.max_set_size=8,16,32,64,128,256,512,1024 \
task.multisets=True \
model.type=mlp

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
paths.log=/tmp/log \
paths.checkpoints=/tmp/checkpoints \
paths.results=/tmp/results \
experiment.results_out=mlp-diff-task-sizes-regression.csv \
experiment.use_gpu=True \
task.label=longest_seq_length,largest_contiguous_sum,max,cardinality \
task.max_set_size=8,16,32,64,128,256,512,1024 \
task.multisets=True \
model.type=mlp

cd $old_dir