import numpy as np
from scipy.linalg import eig


def construct_positive_reciprocal_matrix(A, lambd=0.1):
    """Construct the positive reciprocal matrix following the descriptions
    of Garner and Engelhard

    :param A: _description_
    :type A: _type_
    :return: _description_
    :rtype: _type_
    """
    m, _ = A.shape # By convention, expect A to have (num_items, num_students) shape
    D = np.eye(m)
    
    for i in range(m-1):
        for j in range(i+1, m):
            res_i = A[i, :]
            res_j = A[j, :]
            Bij = max(len(np.where((res_i == 1) & (res_j == 0))[0]), lambd)
            Bji = max(len(np.where((res_j == 1) & (res_i == 0))[0]), lambd)
            D[i, j] = Bji/Bij
            D[j, i] = Bij/Bji
    return D


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



