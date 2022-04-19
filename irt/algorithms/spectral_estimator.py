import numpy as np


def construct_markov_chain(performances, weighted_d=False):
    m = len(performances)
    M = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            if i != j:
                res_i = performances[i, :]
                res_j = performances[j, :]
                Bij = len(np.where((res_i == 1) & (res_j == 0))[0])
                M[i, j] = Bij # Number of students who can solve i but not j
                
    dmax = np.max([np.sum(M[i, :]) for i in range(m)]) + 1
    d = []
    for i in range(m):
        if weighted_d:
            di = np.sum(M[i, :]) + 1
            d.append(di)
            M[i, :] /= di
        else:
            M[i, :] /= dmax
        M[i, i] = 1. - np.sum(M[i, :])
    
    if len(d) == 0:
        d = [dmax for _ in range(m)]
    d = np.array(d)
    return M, d
    

def spectral_estimate(performances, accelerated=True, max_iters=10000, estimate_test=True, regularization=False):
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
    if estimate_test:
        A = performances
    else:
        # Flip 0 and 1 on the performance matrix (A stronger student has more 1's)
        A = performances.T
        zeros_ind = np.where(A == 0)[0]
        ones_ind = np.where(A == 1)[0]
        A[zeros_ind] = 1
        A[ones_ind] = 0
    
    if accelerated:
        M, d = construct_markov_chain(A, weighted_d=True)
    else:
        M, d = construct_markov_chain(A)
    
    m = len(A)        
    pi = np.ones((m,)).T
    for _ in range(max_iters):
        pi_next = (pi @ M)
        pi_next /= np.sum(pi_next)
        if np.linalg.norm(pi_next - pi) < 1e-8:
            pi = pi_next
            break
        pi = pi_next
        
    pi = pi.T/d
    pi /= np.sum(pi)
    return pi


class SpectralEstimator():
    def __init__(self, accelerated=True, max_iters=1000):
        self.accelerated = accelerated
        self.max_iters = max_iters
    
    def fit(self, X):
        # Estimate both set of parameters, shouldn't we estimate test parameters, then use ability estimation method from the
        # Girth
        self.z = spectral_estimate(X, self.accelerated, self.max_iters, True)
        self.w = spectral_estimate(X, self.accelerated, self.max_iters, False)
        return self.z
    
    def predict(self, student_test_pair):
        pred = []
        for (l, i) in student_test_pair:
            pred.append(self.w[l]/(self.w[l] + self.z[i]))
        return pred