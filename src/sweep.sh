#!/bin/bash

python run.py -m \
experiment.random_seed=9,99,999,9999,99999 \
model.accumulator=sum,max \
trainset.label=sum \
validset.label=sum \
testset.label=sum \
trainset.multisets=True \
validset.multisets=True \
testset.multisets=True 