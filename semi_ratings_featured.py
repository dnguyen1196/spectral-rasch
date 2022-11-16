import argparse

parser = argparse.ArgumentParser("Real data experiments on ratings dataset")
parser.add_argument("--dataset", type=str, default="ml_100k", 
                    choices=["ml_100k", "ml_1m", "hetrec_2k", "jester", "ml_10m", "ml_20m", "book_genome", "bx_book", "each_movie"])
parser.add_argument("--seed", type=int, default=119, help="Random seed")
parser.add_argument("--p_train", type=float, default=0.9, help="Training-testing split")
parser.add_argument("--out_folder", type=str, default="./experiment_results/kernel", help="Output folder")
parser.add_argument("--spectral", action="store_true", default=False, help="Run experiments for spectral")


args = parser.parse_args()
out_folder = args.out_folder
dataset = args.dataset
seed = args.seed
p_train = args.p_train

spectral = args.spectral

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
from irt.data import data_loader_featurized as featured_data
from irt.algorithms import RegularizedSpectral, KernelSmoother, NeuralSmoother, LogisticRegression, MultiLayerNeuralNetwork

from irt.algorithms import RegularizedSpectral
import gpytorch as gp

warnings.filterwarnings("ignore")
import torch as th
import os

np.random.seed(seed)

# def save_partial_results():
#     methods = ""
    
#     if spectral:
#         methods += "_spectral"

#     output_file = os.path.join(out_folder, f"{dataset}_sd={seed}_mds={methods}.pkl")

#     # Save results
#     th.save({

#     }, output_file)

def convert_affinity_to_binary(X, Y, by_user=True, rating_data=True):
    A_lr = X @ Y.T
    n, m = A_lr.shape
    average = np.mean(A_lr, 1)
    
    approval = 0 if rating_data else 1
    disapproval = 1 if rating_data else 0
    
    for l in range(n):
        A_l = A_lr[l, :]
        A_lr[l, :] = np.where(A_l < average[l], disapproval * np.ones_like(A_l), approval * np.ones_like(A_l))
    
    return A_lr if by_user else A_lr.T


def get_responses(A, X, Y):
    m, n = A.shape
    INVALID_RESPONSE = -99999
    
    # Convolve X and Y
    XY_features = []
    responses = []
    
    for i in range(m):
        for l in range(n):
            if A[i, l] != INVALID_RESPONSE:
                yi = np.array(np.atleast_2d(Y[i, :]))
                xl = np.array(np.atleast_2d(X[l, :]))
                xy = np.concatenate([yi, xl], 1).T.squeeze()
                XY_features.append(xy)
                responses.append(A[i, l])
    
    return th.tensor(XY_features, dtype=th.float32), th.tensor(responses).float()

# TODO: finish implementation for loading data with preloaded option
# W, H = featured_data.ml_100k()
W, H, average_ratings = th.load("./notebooks/kernel/ml_100k_r=30_WHratings.pkl")
W = W.todense() # TODO: save dense matrix
H = H.todense()
X = W
Y = H.T


n_train = int(len(X) * p_train)
X_train = X[:n_train, :]
X_test = X[n_train:, :]
A_train_full = convert_affinity_to_binary(X_train, Y, False)
A_test = convert_affinity_to_binary(X_test, Y, False)

Z_train, _ = get_responses(A_train_full, X_train, Y)
# Z_val, responses_val = get_responses(A_test, X_test, Y)

INVALID_RESPONSE = -99999
p_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * 0.01
ls_choices = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5]
minibatch = 100
max_epochs = 2000

output_file = os.path.join(out_folder, f"{dataset}_sd={seed}_mds=_spectral.pkl")


results = {
    "W" : W,
    "H" : H,
    "p_array" : p_array,
    "p_train" : p_train,
    "ls_choices" : ls_choices,
    "vanilla_spectral_est": [],
    "kernel_spectral" : [],
    "neural_spectral" : []
}

INVALID_RESPONSE = -99999


for p in p_array:
    print("Doing ", p)
    A_train = featured_data.sparsify(A_train_full, p, seed)
    assignment = np.array([np.all(A_train[:, l] == INVALID_RESPONSE) for l in range(A_train.shape[1])])
    A_train = A_train[:, ~assignment]
    X_train_valid = X_train[~assignment, :]
    
    est_beta = spectral_estimate(A_train)
    results["vanilla_spectral_est"].append(est_beta)

    kernel_spectral_res = {}
    for var_dist in ["MeanField", "Cholesky"]:
        for n_inducing in [100, 200, 300, 400, 500]:
            key = f"{var_dist}-{n_inducing}"
            try:
                kernel = "Matern"
                mean = "Linear"
                kernel_smoother = KernelSmoother(kernel, mean, var_dist)
                kernel_smoother.fit(A_train, X_train_valid, Y, n_inducing_points=n_inducing, lr=0.01, minibatch=minibatch, max_epochs=max_epochs)
                A_smoothed = kernel_smoother.predict_prob(Z_train).reshape(A_train.shape[0], A_train.shape[1]).detach().numpy()
                est_beta = spectral_estimate(A_smoothed, 0)
                kernel_spectral_res[key] = np.copy(est_beta)
            except Exception as e:
                kernel_spectral_res[key] = str(e)
    
    results["kernel_spectral"].append(kernel_spectral_res)
    
    nnet_spectral_res = {}
    try:
        logreg = LogisticRegression(input_dim=X_train_valid.shape[1] + Y.shape[1])
        nnet = MultiLayerNeuralNetwork(input_dim=X_train_valid.shape[1]+Y.shape[1], hidden_layer_sizes=hidden_layer_sizes)
        neural_smoother = NeuralSmoother(logreg)
        neural_smoother.fit(A_train, X_train_valid, Y, lr=0.01, minibatch=minibatch, max_epochs=max_epochs)
        A_smoothed = neural_smoother.predict_prob(Z_train).reshape(A_train.shape[0], A_train.shape[1]).detach().numpy()
        est_beta = spectral_estimate(A_smoothed, 0)
        nnet_spectral_res["logistic"] = np.copy(est_beta)
    except Exception as e:
        nnet_spectral_res["logistic"] = str(e)
    
    
    for hidden_layer_sizes in [ [20, 20], [20, 20, 20], [30, 30]]:
        key = "-".join([str(s) for s in hidden_layer_sizes])
        try:
            nnet = MultiLayerNeuralNetwork(input_dim=X_train_valid.shape[1]+Y.shape[1], hidden_layer_sizes=hidden_layer_sizes)
            neural_smoother = NeuralSmoother(nnet)
            neural_smoother.fit(A_train, X_train_valid, Y, lr=0.01, minibatch=minibatch, max_epochs=max_epochs)
            A_smoothed = neural_smoother.predict_prob(Z_train).reshape(A_train.shape[0], A_train.shape[1]).detach().numpy()
            est_beta = spectral_estimate(A_smoothed, 0)
            nnet_spectral_res[key] = np.copy(est_beta)
        except Exception as e:
            nnet_spectral_res[key] = str(e)
            
    results["neural_spectral"].append(nnet_spectral_res)
    
    # Save results
    th.save(results, output_file)