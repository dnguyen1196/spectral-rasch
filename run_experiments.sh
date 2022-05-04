#!/bin/bash
python=~/miniconda3/envs/conda-py38/bin/python
export PYTHONPATH=$PWD:$PYTHONPATH
out_folder='experiment_results/may/may4/'
seed=119
n_trials=10
include=" --spectral --jmle --cmle "


################## RATINGS DATASET  

$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_100k $include &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_1m  $include &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset hetrec_2k  $include &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_10m $include  &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_20m $include  &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset jester  $include &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset bx_book  $include &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset book_genome $include  &
$python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset each_movie $include  &


# ################### EDUCATION DATASETS 
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset lsat $include &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset uci_student $include &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset grades_three $include &


################# SYNTHETIC DATASETS

# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 1 --student_var 1 --test_var 1 &
# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 1 --student_var 2 --test_var 1   &
# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 0.1 --student_var 1 --test_var 1  &
# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 0.1 --student_var 2 --test_var 1   &
# $python synthetic_experiments.py --out_folder $out_folder --m 500 --p 0.05 --student_var 1 --test_var 1 &
# $python synthetic_experiments.py --out_folder $out_folder --m 500 --p 0.05 --student_var 2 --test_var 1  &