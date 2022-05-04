#!/bin/bash
python=~/miniconda3/envs/conda-py38/bin/python
export PYTHONPATH=$PWD:$PYTHONPATH
out_folder='experiment_results/apr/apr28/'
seed=119
n_trials=100
# "lsat", "uci_student", "grades_three"

# $python real_experiments_education.py --out_folder $out_folder --seed $seed --dataset lsat --n_trials $n_trials --reg uniform &
# $python real_experiments_education.py --out_folder $out_folder --seed $seed --dataset uci_student --n_trials $n_trials --reg uniform  &
# $python real_experiments_education.py --out_folder $out_folder --seed $seed --dataset grades_three --n_trials $n_trials --reg uniform  &

$python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset lsat  &
$python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset uci_student   &
$python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset grades_three   &

