import argparse

parser = argparse.ArgumentParser("Synthetic data experiments")
parser.add_argument("--out_folder", type=str, default="./experiment_results/privacy", help="Output folder")
parser.add_argument("--m", type=int, default=100, help="Number of tests")
parser.add_argument("--student_var", type=float, default=1., help="Student variance")
parser.add_argument("--test_var", type=float, default=1., help="Test variance")
parser.add_argument("--seed", type=int, default=119, help="Random seed")

args = parser.parse_args()
out_folder = args.out_folder
m = args.m
student_var = args.student_var
test_var = args.test_var
seed = args.seed

import numpy as np
import time
from irt.data.rasch import generate_data
from irt.algorithms.spectral_estimator import spectral_estimate
from irt.algorithms.private_spectral_estimator import spectral_estimate_private, randomized_response, subsampl_graph, find_effective_epsilon0_zgauss, find_effective_epsilon0_rr
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

n_trials = 100
INVALID_RESPONSE = -99999
np.random.seed(seed)
betas = np.random.normal(0, test_var, size=(m,))


n_array = [100, 200, 400, 800, 1000, 1500, 2000]
# p_array = [0.2, 0.4, 0.6, 0.8, 1.0]
p_array = [1.0]
epsilon_array = [0.01, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 7.5]

import collections
result_dict = {
    "n_array": n_array,
    "p_array": p_array,
    "epsilon_array" : epsilon_array,
    "m": m,
    "result" : collections.defaultdict(float)
}


def save_partial_results():
    output_file = os.path.join(out_folder, f"synthetic_m={m}_testvar={test_var}_stuvar={student_var}_seed={seed}.th")
    # Save results
    th.save(result_dict, output_file)


overall_delta = 1e-4


for n in n_array:
    thetas = np.random.normal(0, student_var, size=(n,))
    
    for p in p_array:
        # For each p sampling level
        
        for overall_epsilon in epsilon_array:
            # For each desired epsilon level
            
            effective_epsilon0_zgauss = find_effective_epsilon0_zgauss(overall_epsilon, overall_delta)
            effective_epsilon_full = effective_epsilon0_zgauss/np.sqrt(m*(m-1)/2)
            effective_epsilon0_rr = max(find_effective_epsilon0_rr(overall_epsilon, overall_delta, n), overall_epsilon)
            effective_epsilon_lap = overall_epsilon/(m*(m-1))
            
            avg_non_private_err = []
            avg_private_err = []
            avg_rr_err = []
            avg_private_sparse_err = []
            avg_lap_err = []
            
            
            for _ in range(n_trials):
                # Generate data
                data = generate_data(betas, thetas, p)
                rr_data = randomized_response(data, effective_epsilon0_rr)
                
                beta_est = spectral_estimate(data, lambd=1., accelerated=True)
                beta_est_rr = spectral_estimate(rr_data, accelerated=True)
                beta_est_gauss = spectral_estimate_private(data, lambd=1., epsilon=effective_epsilon_full)
                beta_est_lap = spectral_estimate_private(data, lambd=1., mechanism="Lap", epsilon=effective_epsilon_lap)
                
                
                # Sample a sub-graph then run sparse private estimator
                p_edge = (1 + 0.1) * np.log2(m)/m
                subsample_graph = subsampl_graph(m, p_edge)
                effective_epsilon_subgraph = effective_epsilon0_zgauss/np.sqrt(np.sum(subsample_graph)/2)
                beta_est_gauss_sparse = spectral_estimate_private(data, lambd=1., epsilon=effective_epsilon_subgraph, subsample_graph=subsample_graph)

                # Record errors
                err_non_private = np.linalg.norm(beta_est - betas)
                err_private = np.linalg.norm(beta_est_gauss - betas)
                err_private_sparse = np.linalg.norm(beta_est_gauss_sparse - betas)
                err_rr = np.linalg.norm(beta_est_rr - betas)
                err_lap = np.linalg.norm(beta_est_lap - betas)
                
                avg_non_private_err += [err_non_private]
                avg_private_err +=  [err_private]
                avg_rr_err += [err_rr]
                avg_private_sparse_err += [err_private_sparse]
                avg_lap_err += [err_lap]

            result_dict["result"][f"n={n}_p={p}_eps={overall_epsilon}_nonprivate"] = avg_non_private_err
            result_dict["result"][f"n={n}_p={p}_eps={overall_epsilon}_zgauss"] = avg_private_err
            result_dict["result"][f"n={n}_p={p}_eps={overall_epsilon}_zlap"] = avg_lap_err
            result_dict["result"][f"n={n}_p={p}_eps={overall_epsilon}_zgauss_sparse"] = avg_private_sparse_err
            result_dict["result"][f"n={n}_p={p}_eps={overall_epsilon}_rr"] = avg_rr_err
            
            save_partial_results()
            
    
