import argparse
from irt.algorithms import SpectralAlgorithm, generate_polytomous_rasch, pcm_mml, pcm_jml, \
        estimate_abilities_given_difficulties, SpectralEM
from irt.data import poly_rasch_data
from irt.data.poly_rasch_data import get_heldout_ratings
import numpy as np
import torch as th
import os
import time

parser = argparse.ArgumentParser("Mixture of PCM")

parser.add_argument("--out_folder", type=str, default="./experiment_results/polyrasch", help="Output folder for saving experiment results")
parser.add_argument('--seed', type=int, default=119, help='random seed')
parser.add_argument('--dataset', type=str, help='Name of dataset used', default="hetrec_2k", 
                    choices=['hetrec_2k', 'ml_1m', 'ml_20m', 'ml_10m', "each_movie", 
                             "book_genome", "lsat", "uci_student", "grades_three"])
parser.add_argument('--llh', type=str, default="point", help="Likelihood method")
parser.add_argument('--aep', type=str, default="mean", help="Ability estimation method")
parser.add_argument("--max_iters", type=int, default=30, help='Max number of em iters')
parser.add_argument('--m_min', type=int, default=100, help="Min number of ratings threshold for movies")
parser.add_argument('--n_min', type=int, default=100, help="Min number of ratings threshold for users")
parser.add_argument('--init', type=str, default="clustering-pair", help="Initialization method")
parser.add_argument("--auto_sigma", action='store_true', default=False)


args = parser.parse_args()

seed = args.seed
out_folder = args.out_folder
dataset = args.dataset
likelihood_method = args.llh
ability_estimation_method = args.aep
max_iters = args.max_iters
m_min = args.m_min
n_min = args.n_min
init = args.init
auto_sigma = args.auto_sigma

# Save output of algorithm
# Save true parameters value for checking
# ell2 frobenious norm for now
filename = os.path.join(out_folder, f'{dataset}_{seed}_mixture_{init}_{likelihood_method}_{ability_estimation_method}_{m_min}_{n_min}_{max_iters}_{auto_sigma}.th')
err_filename = os.path.join("./logs/", f'{dataset}_{seed}_mixture_{init}_{likelihood_method}_{ability_estimation_method}_{m_min}_{n_min}_{max_iters}_{auto_sigma}_stderr')
import sys
sys.stderr = open(err_filename, 'a+')

import collections
result_dict = collections.defaultdict(list)
C_array = [1,2,3,4,5,6]

# Generate the problem parameters in terms of problems x thresholds
result_dict = vars(args)
dataset_dict = getattr(poly_rasch_data, dataset)(min_n_ratings_movies=m_min, min_n_ratings_users=n_min)
X = dataset_dict['X']
print(f"{dataset}, {X.shape}")
K = int(np.max(X))


X, heldout_ratings = get_heldout_ratings(X, seed)
X, validation_ratings = get_heldout_ratings(X, seed)
result_dict['heldout_ratings'] = heldout_ratings
result_dict['validation_ratings'] = validation_ratings
result_dict['X.shape'] = X.shape
result_dict['C_array'] = C_array
result_dict["K"] = K
result_dict["n"] = X.shape[0]
result_dict["m"] = X.shape[1]
result_dict["auto_sigma"] = auto_sigma

for C in C_array:
    em = SpectralEM(C=C, K=K)
    Z = em.construct_Z_tensor(X, K)
    Z_masked = np.ma.masked_array(Z, mask=(Z==0))
    start = time.time()
    betas_init, thetas_init, labels = em.initialize(X, Z, Z_masked, C, K, ability_estimate_method=ability_estimation_method)
    init_time = time.time() - start
    q_init = em.E_step(X, Z_masked, betas_init, thetas_init, likelihood_method=likelihood_method)
    
    result_dict[f"betas_init_{C}"] = betas_init
    result_dict[f"thetas_init_{C}"] = thetas_init
    result_dict[f"q_init_{C}"] = q_init
    
    try:
        start = time.time()
        betas, thetas, q, iters = em.fit(X, max_iters=max_iters, betas_init=betas_init, 
                                         init=init,
                                         validation_ratings=validation_ratings,
                                         likelihood_method=likelihood_method,
                                         auto_sigma=auto_sigma,
                                         ability_estimate_method=ability_estimation_method)
        result_dict[f"time_{C}"] = time.time() - start
        result_dict[f"betas_{C}"] = betas
        result_dict[f"thetas_{C}"] = thetas
        result_dict[f"q_{C}"] = q
        result_dict[f"iters_{C}"] = iters
        result_dict[f"sigma_{C}"] = em.sigma
        th.save(result_dict, filename)
        
    except Exception as e:
        result_dict[f"betas_{C}"] = betas_init
        result_dict[f"thetas_{C}"] = thetas_init
        result_dict[f'q_{C}'] = q_init
        result_dict[f"time_{C}"] = init_time
        result_dict[f"iters_{C}"] = 0
        result_dict[f"sigma_{C}"] = em.sigma
        th.save(result_dict, filename)

