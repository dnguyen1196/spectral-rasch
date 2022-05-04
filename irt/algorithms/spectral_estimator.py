import numpy as np
from scipy.sparse import csc_matrix
INVALID_RESPONSE = -99999


def construct_markov_chain(performances, lambd=1.):
    m = len(performances)
    
    D = np.ma.masked_where(performances == INVALID_RESPONSE, performances)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    
    A = np.ma.masked_where(performances == INVALID_RESPONSE, np.ones_like(performances))
    B = np.ma.dot(A, A.T)
    
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M)
    
    # Add regularization to the 'missing' entries
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+lambd, M)
    d = []
    for i in range(m):
        di = max(np.sum(B[i, :]), 1)
        d.append(di)
        M[i, :] /= di
        M[i, i] = 1. - np.sum(M[i, :])

    d = np.array(d)
    return M, d
    

def construct_markov_chain_accelerated(performances, lambd=0.1, regularization="uniform"):
    m = len(performances)
    
    D = np.ma.masked_where(performances == INVALID_RESPONSE, performances)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M)
    
    # Add regularization to the 'missing' entries
    if regularization == "uniform":
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M+lambd, M)
    elif regularization == "minimal":
        M = np.where(np.logical_and(np.logical_or((M != 0), (M.T != 0)), 
                                    np.logical_or((M == 0), (M.T == 0))), 
                     M + lambd, M)
    elif regularization == "zero":
        M = np.where(np.logical_and(np.logical_or((M != 0), (M.T != 0)), M == 0), lambd, M)
    else: # No regularization
        M = M
    
    d = []
    for i in range(m):
        di = max(np.sum(M[i, :]), 1)
        d.append(di)
        M[i, :] /= max(d[i], 1)
        M[i, i] = 1. - np.sum(M[i, :])

    d = np.array(d)
    return M, d
    

def spectral_estimate(performances, accelerated=True, max_iters=10000, return_beta=True, lambd=1, regularization="uniform"):
    """Estimate the hidden parameters according to the Rasch model, either for the tests' difficulties
    or the students' abilities. Following the convention of Girth https://eribean.github.io/girth/docs/quickstart/quickstart/
    the rows correspond to the problems and the columns correspond to the students.

    :param performances: _description_
    :type performances: _type_
    :param accelerated: _description_, defaults to False
    :type accelerated: bool, optional
    :param max_iters: _description_, defaults to 10000
    :type max_iters: int, optional
    :param estimate_test: _description_, defaults to True
    :type estimate_test: bool, optional
    :return: _description_
    :rtype: _type_
    """
    assert(regularization in ["uniform", "zero", "minimal", "none"])
    A = performances
    
    if accelerated:
        M, d = construct_markov_chain_accelerated(A, lambd=lambd, regularization=regularization)
    else:
        M, d = construct_markov_chain(A, lambd=lambd)

    M = csc_matrix(M)
    
    m = len(A)        
    pi = np.ones((m,)).T
    for _ in range(max_iters):
        pi_next = (pi @ M)
        pi_next /= np.sum(pi_next)
        if np.linalg.norm(pi_next - pi) < 1e-12:
            pi = pi_next
            break
        pi = pi_next
        
    pi = pi.T/d
    if return_beta:
        beta = np.log(pi)
        beta = beta - np.mean(beta)
        return beta
    
    return pi/np.sum(pi)