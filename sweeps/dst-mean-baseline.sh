#!/bin/bash

#SBATCH --output=/home/smueksch/dsl-repo/results%j.out     
#SBATCH --error=/home/smueksch/dsl-repo/results%j.err

old_dir=$(pwd)
cd ../

python main.py -m \
experiment.random_seed=9,99,999,9999,99999 \
experiment.results_out=dst-mean-baseline-results.csv \
task.label=desperate_student_1_tuple,desperate_student_2_tuple,desperate_student_3_tuple,desperate_student_4_tuple,desperate_student_5_tuple \
model.type=mean_baseline

cd $old_dir