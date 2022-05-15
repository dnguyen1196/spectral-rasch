#!/bin/bash
python=~/miniconda3/envs/conda-py38/bin/python
export PYTHONPATH=$PWD:$PYTHONPATH
out_folder='experiment_results/may/may5/'
seed=119
n_trials=10
# include=" --spectral --mmle --jmle --cmle "
include=" --cmle "
cmle_only=" --cmle "
spectral_only=" --spectral "
mmle_only="--mmle "
jmle_only="--jmle "

################## RATINGS DATASET  

# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_100k $include &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_1m  $include &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset hetrec_2k  $include &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset bx_book  $include &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset each_movie $include  &


# Really large dataset (run individually)
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_10m $spectral_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_10m $mmle_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_10m $cmle_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_10m $jmle_only &

# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_20m $spectral_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_20m $mmle_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_20m $cmle_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_20m $jmle_only &

# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset book_genome $spectral_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset book_genome $mmle_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset book_genome $cmle_only  &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset book_genome $jmle_only  &


# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset ml_1m --cmle &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset hetrec_2k --cmle &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset bx_book --cmle &
# $python real_experiments_ratings_cv.py --out_folder $out_folder --seed $seed --dataset each_movie --cmle  &



# ################### EDUCATION DATASETS 
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset lsat $include &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset uci_student $include &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset grades_three $include &

# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset riiid_small $spectral_only  &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset riiid_small  $mmle_only  &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset riiid_small  $cmle_only  &
# $python real_experiments_education_cv.py --out_folder $out_folder --seed $seed --dataset riiid_small  $jmle_only  &


################# SYNTHETIC DATASETS

# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 1 --student_var 1 --test_var 1 &
# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 1 --student_var 2 --test_var 1   &
# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 0.1 --student_var 1 --test_var 1  &
# $python synthetic_experiments.py --out_folder $out_folder --m 100 --p 0.1 --student_var 2 --test_var 1   &
# $python synthetic_experiments.py --out_folder $out_folder --m 10 --p 0.5 --student_var 1 --test_var 1 &
# $python synthetic_experiments.py --out_folder $out_folder --m 10 --p 0.5 --student_var 2 --test_var 1  &
# $python synthetic_experiments.py --out_folder $out_folder --m 10 --p 1 --student_var 1 --test_var 1 &
# $python synthetic_experiments.py --out_folder $out_folder --m 10 --p 1 --student_var 2 --test_var 1  &