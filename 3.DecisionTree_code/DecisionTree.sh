#!/bin/sh
#PBS -N pro_mon11
#PBS -q cmb -l pmem=120gb -l walltime=15:00:00

cd ./
python DecisionTree.py train.csv label.csv test.csv 10 17

