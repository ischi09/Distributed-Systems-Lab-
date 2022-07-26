#!/bin/bash


#SBATCH --output=/itet-stor/ishamanna/net_scratch/results%j.out     
#SBATCH --error=/itet-stor/ishamanna/net_scratch/results%j.err

old_dir=$(pwd)
cd ../

python main.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.patience=15 \
paths.log=/itet-stor/ishamanna/net_scratch/log \
paths.checkpoints=/itet-stor/ishamanna/net_scratch/checkpoints \
paths.results=/itet-stor/ishamanna/net_scratch/results \
experiment.results_out=master_final_desperate.csv \
task.label=desperate_student_1_tuple,desperate_student_2_tuple,desperate_student_3_tuple,desperate_student_4_tuple,desperate_student_5_tuple \
task.max_set_size=16 \
task.multisets=True \
model.type=deepsets_mlp_sum,deepsets_mlp_fspool,sorted_mlp,mlp,pna,small_set_transformer,mean_baseline

cd $old_dir
