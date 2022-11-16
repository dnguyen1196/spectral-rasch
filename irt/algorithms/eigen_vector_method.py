import numpy as np
from scipy.linalg import eig
INVALID_RESPONSE = -99999
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint



def construct_positive_reciprocal_matrix(A, lambd=0.1):
    """Construct the positive reciprocal matrix following the descriptions
    of Garner and Engelhard

    :param A: _description_
    :type A: _type_
    :return: _description_
    :rtype: _type_
    """
    D = np.ma.masked_where(A == INVALID_RESPONSE, A)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M) # Mij = num(Xi = 1, Xj = 0)
    m = A.shape[0]
    D = np.eye(m)

    for i in range(m-1):
        for j in range(i+1, m):
            Bij = M[i, j]
            Bji = M[j, i]
            if Bij * Bji != 0:
                fij = Bij/(Bij + Bji)
                fji = Bji/(Bij + Bji)
                D[i, j] = fji/fij
                D[j, i] = fij/fji
    return D


def conditional_pairwise(A, step_size=0.1):
    m = A.shape[0]
    D = np.ma.masked_where(A == INVALID_RESPONSE, A)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M) # Yij = num(Xi = 1, Xj = 0)
    
    # Add regularization to the 'missing' entries
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+1, M)
    Y = np.zeros_like(M)
    
    for i in range(m-1):
        for j in range(i+1, m):
            Bij = M[i, j]
            Bji = M[j, i]
            if Bij * Bji != 0:
                Y[i, j] = Bij/(Bij + Bji)
                Y[j, i] = Bji/(Bij + Bji)

    def log_lik_pair(betas):
        exp_beta = np.exp(betas)
        W = np.outer(exp_beta, np.ones((m,))) / (np.outer(exp_beta, np.ones((m,))) + np.outer(np.ones((m,)), exp_beta))
        f = 0.
        grad = np.zeros((m,))
        for i in range(m):
            grad[i] = -np.sum(Y[i, :] * W[i, :]) + np.sum(Y[:, i] * W[:, i])
            f += 1./2 * (np.log(np.sum(Y[i, :] * W[i, :])) + np.log(np.sum(Y[:, i] * W[:, i])))
        return -f, -step_size * grad

    betas0 = np.zeros((m,))
    constraint = LinearConstraint(np.ones((1, m)), 0, 0)
    res = minimize(log_lik_pair, betas0, jac=True, constraints=[constraint])
    betas = res['x']
    return betas


def choppin_method(A, return_beta=True):
    D = construct_positive_reciprocal_matrix(A)
    lnD = np.log(D)
    difficulties = np.mean(lnD, 1)
    # return difficulties
    if return_beta:
        return difficulties - np.mean(difficulties)
    z_est = np.exp(difficulties)
    z_est = z_est / np.sum(z_est)
    return z_est


def garner_method(A, return_beta=True):
    D = construct_positive_reciprocal_matrix(A)
    w, vl, vr = eig(D, left=True, right=True)
    ord = np.argsort(w)[::-1]
    difficulties = vr[:, ord[0]]
    if return_beta:
        return difficulties - np.mean(difficulties)
    
    z_est = np.exp(difficulties)
    z_est = z_est / np.sum(z_est)
    return z_est


def saaty_method(A, max_iters=1000, tol=1e-8, return_beta=True):
    D = construct_positive_reciprocal_matrix(A)
    m = D.shape[0]
    z = np.ones((m,))
    for _ in range(max_iters):
        z_next = D @ z
        z_next /= np.sum(z_next)
        if np.sqrt(np.sum(np.square(z_next - z))) < tol:
            break
        z = z_next
        
    if return_beta:
        difficulties = np.log(z)
        difficulties -= np.mean(difficulties)
        return difficulties
    
    z = z_next
    z /= np.sum(z)
    return z



