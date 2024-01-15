
import argparse
from irt.algorithms import SpectralAlgorithm, generate_polytomous_rasch, pcm_mml, pcm_jml
import numpy as np
import torch as th
import os
from scipy.stats import norm as spnorm
import time

parser = argparse.ArgumentParser()

parser.add_argument("--out_folder", type=str, default="./experiment_results/polyrasch", help="Output folder for saving experiment results")
parser.add_argument('--n_array', nargs='+', help='Array of increasing test takers size', default=[100,200,400,800,1600,3200])
parser.add_argument('--m', type=int, help='Number of tests', default=10)
parser.add_argument('--L', type=int, help='Max number of scores', default=3)
parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
parser.add_argument('--seed', type=int, help='random seed')


parser.add_argument('--student_var', type=float, help='variance of the students ability distribution', default=1.0)
parser.add_argument('--student_mean', type=float, help='Mean of the students ability distribution', default=0.0)
parser.add_argument('--test_var', type=float, help='variance of the test difficulty distribution', default=1.0)
parser.add_argument('--test_mean', type=float, help='Mean of the test difficulty distribution', default=0.0)

parser.add_argument("--spectral", action="store_true", default=False, help="Run experiments for spectral")
parser.add_argument("--mmle", action="store_true", default=False, help="Run experiments for MMLE")
parser.add_argument("--jmle", action="store_true", default=False, help="Run experiments for JMLE")

args = parser.parse_args()

m = args.m
n_array = [int(x) for x in args.n_array]
L = args.L
n_trials = args.n_trials
seed = args.seed
out_folder = args.out_folder
student_var = args.student_var
student_mean = args.student_mean
test_var = args.test_var
test_mean = args.test_mean

do_spectral = args.spectral
do_mmle = args.mmle
do_jmle = args.jmle

# Save output of algorithm
# Save true parameters value for checking
# ell2 frobenious norm for now
filename = os.path.join(out_folder, f'synthetic_{n_array}_{m}_{L}_{student_var}_{student_mean}.th')

import collections
result_dict = collections.defaultdict(list)

np.random.seed(seed)

# Generate the problem parameters in terms of problems x thresholds
result_dict = vars(args)

thetas = np.random.normal(test_mean, test_var, size=(m, L))
result_dict['true_thetas'] = thetas

spectral_by_n = []
mmle_by_n = []
jmle_by_n = []

spectral_time_by_n = []
mmle_time_by_n = []
jmle_time_by_n = []

# For increasing number of samples
for n in n_array:
    # Generate students abilities
    betas = np.random.normal(student_mean, student_var, size=(n,))
    result_dict[f'true_betas_{n}'] = betas
    
    # Generate students' abilities
    # Repeat for n_trials
    spectral = []
    mmle = []
    jmle = []
    
    spectral_time = []
    mmle_time = []
    jmle_time = []
    
    for _ in range(n_trials):
        X = generate_polytomous_rasch(betas, thetas)
        
        # Obtain spectral estimator
        if do_spectral:
            start = time.time()
            estimator = SpectralAlgorithm()
            spectral_thetas = estimator.fit(X)
            spectral.append(spectral_thetas)
            spectral_time.append(time.time() - start)

        # Then obtain JMLE estimator
        if do_jmle:
            start = time.time()
            jmle_thetas = pcm_jml(X)
            jmle.append(jmle_thetas)
            jmle_time.append(time.time() - start)
        
        # Specify the prior distribution for MMLE
        if do_mmle:
            normal_dist = lambda x: spnorm.pdf(x, 0., 1.)
            options = {
                'use_LUT': True,
                'distribution': normal_dist
            }
            start = time.time()
            mmle_thetas = pcm_mml(X, options)
            mmle.append(mmle_thetas)
            mmle_time.append(time.time() - start)
                
    spectral_by_n.append(spectral)
    jmle_by_n.append(jmle)
    mmle_by_n.append(mmle)
    
    spectral_time_by_n.append(spectral_time)
    jmle_time_by_n.append(jmle_time)
    mmle_time_by_n.append(mmle_time)
    
    result_dict['spectral_by_n'] = spectral_by_n
    result_dict['jmle_by_n'] = jmle_by_n
    result_dict['mmle_by_n'] = mmle_by_n
    
    result_dict['spectral_time_by_n'] = spectral_time_by_n
    result_dict['jmle_time_by_n'] = jmle_time_by_n
    result_dict['mmle_time_by_n'] = mmle_time_by_n
    
    th.save(result_dict, filename)
    
th.save(result_dict, filename)