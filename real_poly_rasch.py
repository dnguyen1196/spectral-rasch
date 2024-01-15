
import argparse
from irt.algorithms import SpectralAlgorithm, generate_polytomous_rasch, pcm_mml, pcm_jml, estimate_abilities_given_difficulties
from irt.data import poly_rasch_data
from irt.data.poly_rasch_data import get_heldout_ratings
import numpy as np
import torch as th
import os
from scipy.stats import norm as spnorm
import time

parser = argparse.ArgumentParser()

parser.add_argument("--out_folder", type=str, default="./experiment_results/polyrasch", help="Output folder for saving experiment results")
parser.add_argument('--seed', type=int, help='random seed')
parser.add_argument('--dataset', type=str, help='Name of dataset used', default="hetrec_2k", 
                    choices=['hetrec_2k', 'ml_1m', 'ml_20m', 'ml_10m', "each_movie", 
                             "book_genome", "algebra_05_06", "lsat", "uci_student", "grades_three"])
parser.add_argument("--spectral", action="store_true", default=False, help="Run experiments for spectral")
parser.add_argument("--mmle", action="store_true", default=False, help="Run experiments for MMLE")
parser.add_argument("--jmle", action="store_true", default=False, help="Run experiments for JMLE")


args = parser.parse_args()

seed = args.seed
out_folder = args.out_folder
dataset = args.dataset

do_spectral = args.spectral
do_mmle = args.mmle
do_jmle = args.jmle

methods = ""
if do_spectral:
    methods += "spectral"
if do_mmle:
    methods += "_mmle"
if do_jmle:
    methods += "_jmle"
methods = methods.lstrip("_")

# Save output of algorithm
# Save true parameters value for checking
# ell2 frobenious norm for now
filename = os.path.join(out_folder, f'{dataset}_{seed}_{methods}.th')

import collections
result_dict = collections.defaultdict(list)

np.random.seed(seed)

# Generate the problem parameters in terms of problems x thresholds
result_dict = vars(args)
dataset_dict = getattr(poly_rasch_data, dataset)()

X_orig = dataset_dict['X']
X, heldout_ratings = get_heldout_ratings(X_orig, seed)
X, validation_ratings = get_heldout_ratings(X, seed)

result_dict['heldout_ratings'] = heldout_ratings
result_dict['X.shape'] = X_orig.shape

print(f"dataset = {dataset}, X.shape = {X.shape}")

th.save(result_dict, filename)

# Obtain spectral estimator
if do_spectral:
    start = time.time()
    estimator = SpectralAlgorithm()
    spectral_thetas = estimator.fit(X)
    result_dict['spectral_thetas'] = (spectral_thetas)
    result_dict['spectral_time'] = time.time() - start
    spectral_betas_est = estimate_abilities_given_difficulties(X, spectral_thetas)
    result_dict['spectral_betas'] = spectral_betas_est
    th.save(result_dict, filename)


# Specify the prior distribution for MMLE
mmle_priors = [(-1.0, 1.0),  (0., 1.0), (1.0, 1.0), (-1.0, 2.0),  (0., 2.0), (1.0, 2.0), (-1.0, 3.0), (0., 3.0), (1.0, 3.0)]
result_dict['mmle_priors'] = mmle_priors

if do_mmle:
    result_dict['mmle_thetas'] = []
    result_dict['mmle_time'] = []
    result_dict['mmle_betas'] = []
    # Try the given MMLE prior parameters
    for mu, st in mmle_priors:
        normal_dist = lambda x: spnorm.pdf(x, mu, st)
        options = {
            'use_LUT': True,
            'distribution': normal_dist
        }
        start = time.time()
        mmle_thetas = pcm_mml(X, options)
        mmle_thetas = np.nan_to_num(mmle_thetas, nan=6.)
        
        result_dict['mmle_thetas'].append(mmle_thetas)
        result_dict['mmle_time'].append(time.time() - start)
        mmle_betas = estimate_abilities_given_difficulties(X, mmle_thetas)
        result_dict['mmle_betas'].append(mmle_betas)
        th.save(result_dict, filename)
        
# Then obtain JMLE estimator
if do_jmle:
    start = time.time()
    jmle_thetas = pcm_jml(X)
    jmle_thetas = np.nan_to_num(jmle_thetas, nan=6.)

    result_dict['jmle_thetas'] = jmle_thetas
    result_dict['jmle_time'] = time.time() - start
    jmle_betas = estimate_abilities_given_difficulties(X, jmle_thetas)
    result_dict['jmle_betas'] = jmle_betas
    th.save(result_dict, filename)
