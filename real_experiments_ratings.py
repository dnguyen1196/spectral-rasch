
import argparse

parser = argparse.ArgumentParser("Real data experiments on ratings dataset")
parser.add_argument("--dataset", type=str, default="ml_100k", 
                    choices=["ml_100k", "ml_1m", "hetrec_2k", "jester", "ml_10m", "ml_20m", "book_genome", "bx_book", "each_movie"])
parser.add_argument("--cutoff", type=int, default=25, help="Cut off for removing movies with few ratings")
parser.add_argument("--sigma", type=float, default=1.)
parser.add_argument("--n_trials", type=int, default=1, help="Number of trials")
parser.add_argument("--seed", type=int, default=420, help="Random seed")
parser.add_argument("--out_folder", type=str, default="./output/real_data", help="Output folder")
parser.add_argument("--p_train", type=float, default=0.90, help="Training size")
parser.add_argument("--reg", type=str, default="uniform", help="Regularization type for ASE", choices=["uniform", "zero", "minimal", "none"])

args = parser.parse_args()
out_folder = args.out_folder
dataset = args.dataset
seed = args.seed
sigma = args.sigma
n_trials = args.n_trials
p_train = args.p_train
regularization = args.reg
cutoff = args.cutoff

import numpy as np
from irt.evaluation import eval_utils
from irt.data import data_loader
import matplotlib.pyplot as plt
import time
from irt.data.rasch import generate_data
from irt.algorithms.spectral_estimator import spectral_estimate
from irt.algorithms import conditional_mle
from irt.algorithms import rasch_mml
from irt.evaluation.eval_utils import log_likelihood_heldout, bayesian_auc, pairwise_disagreement_error, top_k_accuracy
import warnings
warnings.filterwarnings("ignore")
import torch as th
import os

auc_ase = []
auc_cmle = []
auc_mmle = []
loglik_ase = []
loglik_cmle = []
loglik_mmle = []
pd_ase = []
pd_cmle = []
pd_mmle = []
time_ase = []
time_cmle = []
time_mmle = []
top_k_ase = []
top_k_cmle = []
top_k_mmle = []

np.random.seed(seed)
trial_seeds = np.random.randint(0, 9999, size=(n_trials,))

# Load data
A, ratings =  getattr(data_loader, dataset)() # Use the default cutoff for each dataset
sorted_ratings = sorted(ratings, key=lambda x: x[1], reverse=True)
true_rank = [item for (item, _, _) in sorted_ratings] # Sort from most popular items

# reverse = True # If the user can't 'solve' a movie, then she likes it

p_test = 1. - p_train

p_array = [0.1,0.2,0.3,0.4,0.5,0.75,1.0]
K_array = [10, 50, 100, 200, 300] if dataset != "jester" else [5, 10, 20, 25, 30]

for i in range(n_trials):
    auc_ase_trial = []
    auc_cmle_trial = []
    auc_mmle_trial = []
    loglik_ase_trial = []
    loglik_cmle_trial = []
    loglik_mmle_trial = []
    pd_ase_trial = []
    pd_cmle_trial = []
    pd_mmle_trial = []
    time_ase_trial = []
    time_cmle_trial = []
    time_mmle_trial = []
    top_k_ase_trial = []
    top_k_cmle_trial = []
    top_k_mmle_trial = []
        
    # For each trial, partition the data
    all_train_data, test_data = eval_utils.partition_data(A, p_train=p_train, p_test=p_test, seed=trial_seeds[i])
    
    for p_sub in p_array:
        # Extract a subset of the columns
        train_data = all_train_data[:, :int(p_sub * all_train_data.shape[1])]

        # Marginal MLE
        start = time.time()
        est_mmle = rasch_mml.rasch_mml(train_data, return_beta=True)
        time_mmle_trial += [(time.time() - start)]
        loglik_mmle_trial += [log_likelihood_heldout(est_mmle, test_data, sigma)]
        auc_mmle_trial += [bayesian_auc(est_mmle, test_data, sigma)]
        est_rank_mmle = np.argsort(est_mmle)[::-1]
        top_k_mmle_trial += [
            [top_k_accuracy(true_rank, est_rank_mmle, K) for K in K_array]
        ]
        
        # Accelerated spectral method
        try:
            start = time.time()
            est_ase = spectral_estimate(train_data, lambd=1., regularization=regularization)
            time_ase_trial += [(time.time() - start)]
            loglik_ase_trial += [log_likelihood_heldout(est_ase, test_data, sigma)]
            auc_ase_trial += [bayesian_auc(est_ase, test_data, sigma)]
            est_rank_ase = np.argsort(est_ase)[::-1]
            top_k_ase_trial += [
                [top_k_accuracy(true_rank, est_rank_ase, K) for K in K_array]
            ]
        except Exception as e:
            start = time.time()
            est_ase = spectral_estimate(train_data, lambd=1., regularization="uniform") # If error -> try the uniform regularization
            time_ase_trial += [(time.time() - start)]
            loglik_ase_trial += [log_likelihood_heldout(est_ase, test_data, sigma)]
            auc_ase_trial += [bayesian_auc(est_ase, test_data, sigma)]
            est_rank_ase = np.argsort(est_ase)[::-1]
            top_k_ase_trial += [
                [top_k_accuracy(true_rank, est_rank_ase, K) for K in K_array]
            ]
        
    auc_ase.append(auc_ase_trial)
    auc_cmle.append(auc_cmle_trial)
    auc_mmle.append(auc_mmle_trial)
    
    loglik_ase.append(loglik_ase_trial)
    loglik_cmle.append(loglik_cmle_trial)
    loglik_mmle.append(loglik_mmle_trial)
    
    pd_ase.append(pd_ase_trial)
    pd_cmle.append(pd_cmle_trial)
    pd_mmle.append(pd_mmle_trial)
    
    time_ase.append(time_ase_trial)
    time_cmle.append(time_cmle_trial)
    time_mmle.append(time_mmle_trial)
    
    top_k_ase.append(top_k_ase_trial)
    top_k_cmle.append(top_k_cmle_trial)
    top_k_mmle.append(top_k_mmle_trial)


output_file = os.path.join(out_folder, f"{dataset}_m={A.shape[0]}_{seed}_{p_train}_{n_trials}_mmle+ase-{regularization}.th")

# Save results
th.save({
        "sigma" : sigma,
        "seed" : seed,
        "dataset" : dataset,
        
        "auc_ase" : auc_ase,
        "loglik_ase" : loglik_ase,
        "pd_ase" : pd_ase,
        "top_k_ase" : top_k_ase,
        
        "auc_mmle" : auc_mmle,
        "loglik_mmle" : loglik_mmle,
        "pd_mmle" : pd_mmle,
        "top_k_mmle" : top_k_mmle,
        
        "auc_cmle" : auc_cmle,
        "loglik_cmle" : loglik_cmle,
        "pd_cmle" : pd_cmle,
        "top_k_cmle" : time_cmle,
        
        "time_ase" : time_ase,
        "time_cmle" : time_cmle,
        "time_mmle" : time_mmle,
        
        "p_train" : p_train,
        "p_array" : p_array,
        "K_array" : K_array,
        
        "trial_seeds" : trial_seeds,
        
    }, output_file)