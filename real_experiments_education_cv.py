
import argparse

parser = argparse.ArgumentParser("Real data experiments on EDUCATION datasets with cross validation")
parser.add_argument("--dataset", type=str, default="ml_100k", choices=["lsat", "uci_student", "grades_three", "riiid", "riiid_small"])
parser.add_argument("--seed", type=int, default=420, help="Random seed")
parser.add_argument("--out_folder", type=str, default="./output/real_data", help="Output folder")
parser.add_argument("--p_train", type=float, default=0.80, help="Training size")
parser.add_argument("--reg", type=str, default="uniform", help="Regularization type for ASE", choices=["uniform", "zero", "minimal", "none"])
parser.add_argument("--cmle", action="store_true", default=False, help="Run experiments for cmle")
parser.add_argument("--jmle", action="store_true", default=False, help="Run experiments for jmle")
parser.add_argument("--spectral", action="store_true", default=False, help="Run experiments for spectral")
parser.add_argument("--mmle", action="store_true", default=False, help="Run experiments for mmle")

args = parser.parse_args()
out_folder = args.out_folder
dataset = args.dataset
seed = args.seed
p_train = args.p_train
regularization = args.reg
include_cmle = args.cmle
include_jmle = args.jmle
include_spectral = args.spectral
include_mmle = args.mmle


import numpy as np
from irt.evaluation import eval_utils
from irt.data import data_loader
import matplotlib.pyplot as plt
import time
from irt.data.rasch import generate_data
from irt.algorithms.spectral_estimator import spectral_estimate
from irt.algorithms import conditional_mle
from irt.algorithms import rasch_mml
from irt.algorithms import joint_mle
from irt.evaluation.eval_utils import log_likelihood_heldout, bayesian_auc
import warnings
warnings.filterwarnings("ignore")
import torch as th
import os
from scipy.stats import norm
from sklearn.model_selection import KFold


np.random.seed(seed)

# Load data
A =  getattr(data_loader, dataset)() 
p_test = 1. - p_train

prior_dist = [(0, 0.5), (0, 1.), (0, 1.5), (-1, 0.5), (-1, 1.), (-1, 1.5), (1, 0.5), (1, 1.), (1, 1.5), (-2, 0.5), (-2, 1), (-2, 1.5), (2, 0.5), (2, 1.), (2, 1.5)]

# Partition the data into train and test set using the global seed
A_train_all, test_data = eval_utils.partition_data(A, p_train=p_train, p_test=p_test, seed=seed)
binary_responses = eval_utils.extract_binary_responses(A_train_all)


def cross_validation_folds(A_train_all, n_folds=10):
    folds = []
    _, n = A_train_all.shape
    # Simply return 10 sections
    x = np.ones((n,))
    folds = KFold(n_splits=n_folds)
    return folds.split(x)


if include_spectral:
    ################################### Spectral ##############################
    m, n_train = A_train_all.shape
    train_cv_split = 0.8
    start = time.time() # Train once
    est_spectral = spectral_estimate(A_train_all, lambd=0., regularization="uniform")
    # est_spectral = spectral_estimate(A_train_all[:, :int(n_train * train_cv_split)], lambd=1., regularization="uniform")
    time_spectral = time.time() - start
    loglik_spectral = []
    p_estimates_spectral = []

    # binary_response_cv = eval_utils.extract_binary_responses(A_train_all[:, int(n_train*train_cv_split):])
    prior_dist_spectral = prior_dist
    for mu, sigma in prior_dist_spectral:
        start = time.time()
        p_estimate_spectral = eval_utils.quadrature_p_response(est_spectral, sigma, mu)
        p_estimates_spectral.append(p_estimate_spectral)
        loglik_spectral.append(log_likelihood_heldout(p_estimate_spectral, binary_responses))
        # loglik_spectral.append(log_likelihood_heldout(p_estimate_spectral, binary_response_cv)) # Test on the CV fold

    # Pick the best variance for highest log likelihood (train data) then evaluate on test data
    # mu_best_spectral, sigma_best_spectral = prior_dist[np.argmax(loglik_spectral)]
    mu_best_spectral, sigma_best_spectral = prior_dist_spectral[np.argmax(loglik_spectral)]
    p_estimate_spectral = p_estimates_spectral[np.argmax(loglik_spectral)]
    # Then retrain on the whole dataset
    # est_spectral = spectral_estimate(A_train_all, lambd=1., regularization="uniform")

    test_loglik_spectral = log_likelihood_heldout(p_estimate_spectral, test_data)
    test_auc_spectral = bayesian_auc(p_estimate_spectral, test_data)
    est_rank_spectral = np.argsort(est_spectral)[::-1]

    print("Save results for spectral")
    output_file = os.path.join(out_folder, f"{dataset}_m={A.shape[0]}_{seed}_{p_train}_spectral.th")

    # Save results
    th.save({
            "est_spectral" : est_spectral,
            "sigma_spectral" : sigma_best_spectral,
            "mu_spectral" : mu_best_spectral,
            "auc_spectral" : test_auc_spectral,
            "loglik_spectral" : test_loglik_spectral,
            "time_spectral" : time_spectral,

            "seed" : seed,
            # "students_vars" : prior_dist,
            "students_vars" : prior_dist_spectral,
            
        }, output_file)


if include_mmle:
    ################################### MMLE ##############################
    loglik_mmle = []
    estimates_mmle = []
    p_estimates_mmle = []
    # Try for each prior distribution
    time_mmle = 0.

    for mu, sigma in prior_dist:
        distribution = lambda x: norm.pdf(x, mu, sigma)
        options = {"distribution" : distribution}
        start = time.time()
        est_mmle = rasch_mml.rasch_mml(A_train_all, return_beta=True, options=options)
        time_mmle += 1./len(prior_dist) * (time.time() - start)
        start = time.time()
        p_estimate = eval_utils.quadrature_p_response(est_mmle, sigma, mu) # Estimate p response
        loglik_mmle.append(log_likelihood_heldout(p_estimate, binary_responses))
        estimates_mmle.append(est_mmle)
        p_estimates_mmle.append(p_estimate)

    loglik_mmle = np.array(loglik_mmle)

    # Pick the best loglikelihood and evaluate on test data
    mu_best_mmle, sigma_best_mmle = prior_dist[np.argmax(loglik_mmle)]
    est_mmle = estimates_mmle[np.argmax(loglik_mmle)]

    p_estimate_mmle = p_estimates_mmle[np.argmax(loglik_mmle)]
    test_loglik_mmle = log_likelihood_heldout(p_estimate_mmle, test_data)
    start = time.time()
    test_auc_mmle = bayesian_auc(p_estimate_mmle, test_data)

    start = time.time()
    est_rank_mmle = np.argsort(est_mmle)[::-1]

    start = time.time()

    print("Save results for MMLE")
    output_file = os.path.join(out_folder, f"{dataset}_m={A.shape[0]}_{seed}_{p_train}_MMLE.th")

    # Save results
    th.save({
            "est_mmle" : est_mmle,
            "sigma_mmle" : sigma_best_mmle,
            "mu_mmle" : mu_best_mmle,
            "auc_mmle" : test_auc_mmle,
            "loglik_mmle" : test_loglik_mmle,
            "time_mmle" : time_mmle,

            "seed" : seed,
            "students_vars" : prior_dist,
        }, output_file)


################################### Joint MLE #######################

if include_jmle:
    start = time.time() # Train once
    _, est_jmle = joint_mle.rasch_jml(A_train_all)
    est_jmle = est_jmle - np.mean(est_jmle) # Normalize parameters
    time_jmle = time.time() - start
    loglik_jmle = []
    p_estimates_jmle = []

    for mu, sigma in prior_dist:
        start = time.time()
        p_estimate_jmle = eval_utils.quadrature_p_response(est_jmle, sigma, mu)
        p_estimates_jmle.append(p_estimate_jmle)
        loglik_jmle.append(log_likelihood_heldout(p_estimate_jmle, binary_responses))

    # Pick the best variance for highest log likelihood (train data) then evaluate on test data
    mu_best_jmle, sigma_best_jmle = prior_dist[np.argmax(loglik_jmle)]
    p_estimate_jmle = p_estimates_jmle[np.argmax(loglik_jmle)]
    test_loglik_jmle = log_likelihood_heldout(p_estimate_jmle, test_data)
    test_auc_jmle = bayesian_auc(p_estimate_jmle, test_data)
    est_rank_jmle = np.argsort(est_jmle)[::-1]

    # Since CMLE might take a long time, save current results first
    output_file = os.path.join(out_folder, f"{dataset}_m={A.shape[0]}_{seed}_{p_train}_JMLE.th")

    # Save results
    th.save({
            "est_jmle" : est_jmle,
            "sigma_jmle" : sigma_best_jmle,
            "mu_jmle" : mu_best_jmle,
            "auc_jmle" : test_auc_jmle,
            "loglik_jmle" : test_loglik_jmle,
            "time_jmle" : time_jmle,

            "seed" : seed,
            "students_vars" : prior_dist,
            
        }, output_file)


################################### Conditional MLE #######################
if include_cmle:
    start = time.time()
    est_cmle = conditional_mle.rasch_conditional(A_train_all, return_beta=True)
    time_cmle = time.time() - start
    loglik_cmle = []
    p_estimates_cmle = []

    for mu, sigma in prior_dist:
        p_estimate_cmle = eval_utils.quadrature_p_response(est_cmle, sigma, mu)
        p_estimates_cmle.append(p_estimate_cmle)
        loglik_cmle.append(log_likelihood_heldout(p_estimate_cmle, binary_responses))

    mu_best_cmle, sigma_best_cmle = prior_dist[np.argmax(loglik_cmle)]
    p_estimate_cmle = p_estimates_cmle[np.argmax(loglik_cmle)]
    test_loglik_cmle = log_likelihood_heldout(p_estimate_cmle, test_data)
    test_auc_cmle = bayesian_auc(p_estimate_cmle, test_data)
    est_rank_cmle = np.argsort(est_cmle)[::-1]


    #################################### Save output (ALL methods)
    output_file = os.path.join(out_folder, f"{dataset}_m={A.shape[0]}_{seed}_{p_train}_CMLE.th")

    # Save results
    th.save({
            "est_cmle" : est_cmle,
            "sigma_cmle" : sigma_best_cmle,
            "mu_cmle" : mu_best_cmle,
            "auc_cmle" : test_auc_cmle,
            "loglik_cmle" : test_loglik_cmle,
            "time_cmle" : time_cmle,

            "seed" : seed,
            "students_vars" : prior_dist,
            "p_train" : p_train,
        }, output_file)