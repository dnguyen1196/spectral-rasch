import argparse

parser = argparse.ArgumentParser("Synthetic data experiments")
parser.add_argument("--out_folder", type=str, default="./output/synthetic_data", help="Output folder")
parser.add_argument("--m", type=int, default=100, help="Number of tests")
parser.add_argument("--student_var", type=float, default=1., help="Student variance")
parser.add_argument("--test_var", type=float, default=1., help="Test variance")
parser.add_argument("--p", type=float, default=0.1, help="p")

args = parser.parse_args()
out_folder = args.out_folder
m = args.m
student_var = args.student_var
test_var = args.test_var
p = args.p

import numpy as np
import time
from irt.data.rasch import generate_data
from irt.algorithms.spectral_estimator import spectral_estimate
from irt.algorithms import conditional_mle, eigen_vector_method
from irt.algorithms import rasch_mml
from irt.algorithms import joint_mle

from scipy.stats import norm
import os
import torch as th

def ell_2_error(beta, betah):
    beta_norm = beta - np.mean(beta) # Normalize
    betah_norm = betah - np.mean(betah) # Normalize
    return np.linalg.norm(beta_norm - betah_norm)

errors_spectral_arr = []
errors_cmle_arr = []
errors_mmle_arr = []
errors_jmle_arr = []
errors_choppin_arr = []
errors_saaty_arr = []
errors_pair_arr = []

time_spectral_arr = []
time_cmle_arr = []
time_mmle_arr = []
time_jmle_arr = []
time_choppin_arr = []
time_saaty_arr = []
time_pair_arr = []

n_array = [50, 100, 500, 1000, 2500, 5000]
n_trials = 100
betas = np.random.normal(0, test_var, size=(m,))

for n in n_array:
    thetas = np.random.normal(0, student_var, size=(n,))

    error_spectral = []
    error_cmle = []
    error_mmle = []
    error_jmle = []
    error_choppin = []
    error_saaty = []
    error_pair = []

    time_spectral = []
    time_cmle = []
    time_mmle = []
    time_jmle = []
    time_choppin = []
    time_saaty = []
    time_pair = []
    
    for _ in range(n_trials):
        # Generate data
        data = generate_data(betas, thetas, p)
        
        # Conditional MLE
        start = time.time()
        est_cmle = conditional_mle.rasch_conditional(data, return_beta=True)
        time_cmle += [(time.time() - start)]
        error_cmle += [ell_2_error(betas, est_cmle)]
        
        # Marginal MLE
        start = time.time()
        est_mmle = rasch_mml.rasch_mml(data, return_beta=True)
        time_mmle += [(time.time() - start)]
        error_mmle += [ell_2_error(betas, est_mmle)]

        # Joint MLE
        start = time.time()
        _, est_jmle = joint_mle.rasch_jml(data)
        time_jmle += [(time.time() - start)]
        error_jmle += [ell_2_error(betas, est_jmle)]

        # Spectral method
        start = time.time()
        est_ase = spectral_estimate(data, lambd=0.001, regularization="uniform")
        time_spectral += [(time.time() - start)]
        error_spectral += [ell_2_error(betas, est_ase)]

        # Choppin method
        start = time.time()
        est_choppin = eigen_vector_method.choppin_method(data, return_beta=True)
        time_choppin += [(time.time() - start)]
        error_choppin += [ell_2_error(betas, est_choppin)]
                
        # Saaty's method
        start = time.time()
        est_saaty = eigen_vector_method.saaty_method(data, return_beta=True)
        time_saaty += [(time.time() - start)]
        error_saaty += [ell_2_error(betas, est_saaty)]
        
        # Pairwise method
        start = time.time()
        est_pair = eigen_vector_method.conditional_pairwise(data, 0.1)
        time_pair += [(time.time() - start)]
        error_pair += [ell_2_error(betas, est_pair)]     
        
    errors_spectral_arr.append(error_spectral)
    errors_cmle_arr.append(error_cmle)
    errors_mmle_arr.append(error_mmle)
    errors_jmle_arr.append(error_jmle)
    errors_choppin_arr.append(error_choppin)
    errors_saaty_arr.append(error_saaty)
    errors_pair_arr.append(error_pair)

    time_spectral_arr.append(time_spectral)
    time_cmle_arr.append(time_cmle)
    time_mmle_arr.append(time_mmle)
    time_jmle_arr.append(time_jmle)
    time_choppin_arr.append(time_choppin)
    time_saaty_arr.append(time_saaty)
    time_pair_arr.append(time_pair)
    
    

#################################### Save output (ALL methods)
output_file = os.path.join(out_folder, f"synthetic_m={m}_p={p}_testvar={test_var}_stuvar={student_var}.th")

# Save results
th.save({
        "n_array" : n_array,
        "p" : p,
        "test_var" : test_var,
        "student_var" : student_var,

        "errors_spectral" : errors_spectral_arr, 
        "errors_mmle" : errors_mmle_arr,
        "errors_cmle" : errors_cmle_arr,
        "errors_jmle" : errors_jmle_arr,
        "errors_choppin" : errors_choppin_arr,
        "errors_saaty" : errors_saaty_arr,
        "errors_pair" : errors_pair_arr,

        "time_spectral" : time_spectral_arr,
        "time_mmle" : time_mmle_arr,
        "time_cmle" : time_cmle_arr,
        "time_jmle" : time_jmle_arr,
        "time_choppin" : time_choppin_arr,
        "time_saaty" : time_saaty_arr,
        "time_pair" : time_pair_arr,
        
    }, output_file)