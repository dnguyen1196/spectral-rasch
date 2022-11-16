
import numpy as np
from scipy.sparse import csc_matrix
INVALID_RESPONSE = -99999


def message_passing(data, max_iters=200, eps=1e-3, return_user_idx=False):
    # Remove bad response pattern
    D = np.ma.masked_equal(data, INVALID_RESPONSE)
    variances = np.ma.var(D, 0) # Check variance for the users
    good_response = np.ma.where(variances != 0)[0]

    A = D[:, good_response]
    # Jointly estimate the user and item parameters by running fixed point estimate.
    # A = data
    m, n = A.shape
    # w = np.ones((m,)) * 1./(m+n)
    # z = np.ones((n,)) * 1./(m+n)

    rasch_e = np.zeros((m+n,)) #  np.random.uniform(size=(n+m))
    # rasch_e = rasch_e/np.sum(rasch_e)

    for _ in range(max_iters):
        rasch_prev = np.copy(rasch_e)

        for i in range(n):
            sum_i = 0
            for j in range(m):
                sum_i += A[j, i]*(np.exp(rasch_e[i]) + np.exp(rasch_e[n+j]))
            rasch_e[i] = np.log(sum_i/m)

        rasch_e = rasch_e - np.mean(rasch_e)

        for j in range(m): # Items
            sum_j = 0
            for i in range(n):
                sum_j += (1-A[j,i])*(np.exp(rasch_e[i]) + np.exp(rasch_e[n+j]))
            rasch_e[n+j] = np.log(sum_j/n)

        rasch_e = rasch_e - np.mean(rasch_e)

        if np.linalg.norm(rasch_prev - rasch_e) < eps: # Check for convergence
            break


    """
    for it in range(max_iter):
        #     rasch_prev = rasch_e
        for i in range(n):
            sum_i = 0
            for j in range(m):
                sum_i += samples[i,j]*(rasch_e[i] + rasch_e[n+j])
            rasch_e[i] = sum_i/m
        rasch_e = rasch_e/np.sum(rasch_e)

        for j in range(m):
            sum_j = 0
            for i in range(n):
                sum_j += (1-samples[i,j])*(rasch_e[i] + rasch_e[n+j])
            rasch_e[n+j] = sum_j/n
        rasch_e = rasch_e/np.sum(rasch_e)
    """

    betas = rasch_e[n:]
    thetas = rasch_e[:n]
    shift = np.mean(betas)

    return thetas-shift, betas-shift