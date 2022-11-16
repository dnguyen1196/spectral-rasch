import argparse

parser = argparse.ArgumentParser("Real data experiments")
parser.add_argument("--out_folder", type=str, default="./output/synthetic_data", help="Output folder")
parser.add_argument("--m", type=int, default=200, help="Number of items for subsampling")
parser.add_argument("--dataset", type=str, default="ml_100k", 
                    choices=["ml_100k", "ml_1m", "hetrec_2k", "jester", "ml_10m", "ml_20m", "book_genome", "bx_book", "each_movie", "lsat", "uci_student", "grades_three"])
parser.add_argument("--seed", default=119, help="Random seed for reproducibility")
parser.add_argument('--sparse_p', nargs='+', help='Probability of sampling an edge', default=[0.0, 0.2, 0.3, 0.4, 0.5], type=float)
parser.add_argument("--n_trials", default=100, type=int, help="Number of trials")


args = parser.parse_args()
out_folder = args.out_folder
m_subsample = args.m
dataset = args.dataset
seed = args.seed
sparse_p = args.sparse_p
n_trials = args.n_trials

import numpy as np
import time
from irt.data.rasch import generate_data
from irt.algorithms.spectral_estimator import spectral_estimate
from irt.data import data_loader

from irt.algorithms.private_spectral_estimator import spectral_estimate_private, randomized_response, subsampl_graph, find_effective_epsilon0_zgauss, find_effective_epsilon0_rr, INVALID_RESPONSE

from irt.data import data_loader_featurized as featured_data


from irt.algorithms import conditional_mle, eigen_vector_method, pairwise_mle
from irt.algorithms import rasch_mml
from irt.algorithms import joint_mle
from irt.algorithms import bayesian_1pl

from scipy.stats import norm
import os
import torch as th

def ell_2_error(beta, betah):
    beta_norm = beta - np.mean(beta) # Normalize
    betah_norm = betah - np.mean(betah) # Normalize
    return np.linalg.norm(beta_norm - betah_norm)

errors_spectral_arr = []
errors_private_spectral_arr = []
errors_rr_spectral_arr = []
errors_private_heuristic_spectral_arr = []


education_datasets = ["lsat", "uci_student", "grades_three"]

np.random.seed(seed)

if dataset in education_datasets:
    A =  getattr(data_loader, dataset)()
else:
    W, H =  getattr(featured_data, dataset)()
    A = W @ H
    A = np.asarray(A.T)
    m, n = A.shape
    avg_scores = np.mean(A, 0)   
    for l in range(n):
        A[:, l] = np.where(A[:, l] > avg_scores[l], 0, 1)
    
    # Pick a random set of rows (items)
    selected_items = np.random.permutation(m)[:min(m, m_subsample)]
    A = A[selected_items, :] # Pick these random rows

m, n = A.shape
epsilon_array = [0.01, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0]
# p_array = [0.5, 0.8, 1.0] if dataset in education_datasets else [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
p_array = [1.0]


import collections
result_dict = {
    "epsilon_array" : epsilon_array,
    "p_array" : p_array,
    "result" : collections.defaultdict(float),
    "m" : m,
    "n" : n,
    "sparse_p" : sparse_p,
}

def sample_full_responses(A, p):
    m, n = A.shape
    response = np.random.rand(m, n)
    A_sampled = np.where(response < p, A, INVALID_RESPONSE)
    return A_sampled

def save_partial_results():
    output_file = os.path.join(out_folder, f"{dataset}_m={m}_n={n}_seed={seed}.th")
    # Save results
    th.save(result_dict, output_file)

full_data = A
assert(not np.any(np.isnan(full_data)))

beta_non_private = spectral_estimate(full_data, lambd=1., accelerated=True)
m, n = full_data.shape
overall_delta = 1e-4

for p in p_array:
    # For each p sampling level
    # First sample the graph
    sampled_data = sample_full_responses(full_data,  p)

    for overall_epsilon in epsilon_array:
        # For each desired epsilon level
        effective_epsilon0_zgauss = find_effective_epsilon0_zgauss(overall_epsilon, overall_delta)
        effective_epsilon_full = effective_epsilon0_zgauss/np.sqrt(m*(m-1))
        effective_epsilon0_rr = max(find_effective_epsilon0_rr(overall_epsilon, overall_delta, n), overall_epsilon)
        effective_epsilon_lap = overall_epsilon/(m*(m-1))
        
        err_private = []
        err_rr = []
        err_private_sparse = []
        err_lap = []
        
        for _ in range(n_trials):
            # Generate data
            rr_data = randomized_response(sampled_data, effective_epsilon0_rr)
            beta_est_rr = spectral_estimate(rr_data, accelerated=True)
            
            # Discrete Gaussian mechanism
            beta_est_gauss = spectral_estimate_private(sampled_data, lambd=1., epsilon=effective_epsilon_full)
            beta_est_lap = spectral_estimate_private(sampled_data, lambd=1., mechanism="Lap", epsilon=effective_epsilon_lap)

            all_private_sparse = []
            # Sample a sub-graph then run sparse private estimator
            for p_edge in sparse_p:
                p_edge = 2 * np.log2(m)/m if np.abs(p_edge - 0) < 1e-6 else p_edge
                subsample_graph = subsampl_graph(m, p_edge)
                effective_epsilon_subgraph = effective_epsilon0_zgauss/np.sqrt(np.sum(subsample_graph))
                beta_est_gauss_sparse = spectral_estimate_private(sampled_data, lambd=1., epsilon=effective_epsilon_subgraph, subsample_graph=subsample_graph)
                all_private_sparse += [np.linalg.norm(beta_est_gauss_sparse - beta_non_private)]

            err_private += [np.linalg.norm(beta_est_gauss - beta_non_private)]
            err_rr += [np.linalg.norm(beta_est_rr - beta_non_private)]
            err_private_sparse += [all_private_sparse]
            err_lap += [np.linalg.norm(beta_est_lap - beta_non_private)]

        result_dict["result"][f"eps={overall_epsilon}_p={p}_zgauss"] = err_private
        result_dict["result"][f"eps={overall_epsilon}_p={p}_zlap"] = err_lap
        result_dict["result"][f"eps={overall_epsilon}_p={p}_zgauss_sparse"] = err_private_sparse
        result_dict["result"][f"eps={overall_epsilon}_p={p}_rr"] = err_rr
        save_partial_results()
            
    
