#!/bin/bash

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
model.accumulator=sum,max \
set_vals.label=sum,max,cardinality,mode,mean,longest_seq_length,largest_contiguous_sum,largest_pair_sum,largest_triple_sum \
set_vals.multisets=True,False \