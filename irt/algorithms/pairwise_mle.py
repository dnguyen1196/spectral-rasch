import numpy as np
import abc


class ConvergenceTest(metaclass=abc.ABCMeta):

    """Abstract base class for convergence tests.
    Convergence tests should implement a single function, `__call__`, which
    takes a parameter vector and returns a boolean indicating whether or not
    the convergence criterion is met.
    """

    @abc.abstractmethod
    def __call__(self, params, update=True):
        """Test whether convergence criterion is met.
        The parameter `update` controls whether `params` should replace the
        previous parameters (i.e., modify the state of the object).
        """


class NormOfDifferenceTest(ConvergenceTest):

    """Convergence test based on the norm of the difference vector.
    This convergence test computes the difference between two successive
    parameter vectors, and declares convergence when the norm of this
    difference vector (normalized by the number of items) is below `tol`.
    """

    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params, update=True):
        params = np.asarray(params) - np.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False
        dist = np.linalg.norm(self._prev_params - params, ord=self._ord)
        if update:
            self._prev_params = params
        return dist <= self._tol * len(params)


def log_transform(weights):
    """Transform weights into centered log-scale parameters."""
    params = np.log(weights)
    return params - params.mean()



def exp_transform(params):
    """Transform parameters into exp-scale weights."""
    weights = np.exp(np.asarray(params) - np.mean(params))
    return (len(weights) / weights.sum()) * weights



def _mm(n_items, data, initial_params, alpha, max_iter, tol, mm_fun):
    """
    Iteratively refine MM estimates until convergence.
    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    if initial_params is None:
        params = np.zeros(n_items)
    else:
        params = initial_params
    converged = NormOfDifferenceTest(tol=tol, order=1)
    for _ in range(max_iter):
        nums, denoms = mm_fun(n_items, data, params)
        params = log_transform((nums + alpha) / (denoms + alpha))
        if converged(params):
            return params
    return params


def _mm_pairwise(n_items, data, params):
    """Inner loop of MM algorithm for pairwise data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for winner, loser, num_wins in data: # To adapt to our setting: data should consist of (i, j, Yji)
        wins[winner] += num_wins
        val = 1.0 / (weights[winner] + weights[loser])
        denoms[winner] += val * num_wins
        denoms[loser] += val * num_wins
    return wins, denoms


def cmle_pairwise(
        A, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):

    m = A.shape[0]
    D = np.ma.masked_where(A == -99999, A)
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    np.fill_diagonal(M, 0)
    np.nan_to_num(M, False)
    M = np.round(M) # Yij = num(Xi = 1, Xj = 0)
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+1, M)
    
    i_indices, j_indices = np.nonzero(M)
    data = [(i, j, M[j, i]) for i, j in list(zip(i_indices, j_indices))]

    betas = _mm(
            m, data, initial_params, alpha, max_iter, tol, _mm_pairwise)

    return betas - np.mean(betas)