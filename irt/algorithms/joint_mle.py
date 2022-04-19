
import numpy as np
from scipy.optimize import fmin_slsqp, fminbound
from scipy.special import expit

from girth.utilities import (convert_responses_to_kernel_sign,
    mml_approx, trim_response_set_and_counts, validate_estimation_options)


def _jml_abstract(dataset, _item_min_func,
                  discrimination=1, max_iter=25):
    """ Defines common framework for joint maximum likelihood
        estimation in dichotomous models."""
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    n_items, _ = unique_sets.shape

    # Use easy model to seed guess
    alphas = np.full((n_items,), discrimination,
                     dtype='float')  # discrimination
    betas = mml_approx(dataset, alphas)  # difficulty

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    n_takers = unique_sets.shape[1]
    the_sign = convert_responses_to_kernel_sign(unique_sets)
    thetas = np.zeros((n_takers,))

    for iteration in range(max_iter):
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            # pylint: disable=cell-var-from-loop
            scalar = the_sign[:, ndx] * alphas

            def _theta_min(theta):
                otpt = np.exp(scalar * (theta - betas))

                return np.log1p(otpt).sum()

            # Solves for the ability for each person
            thetas[ndx] = fminbound(_theta_min, -6, 6)

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)

        #####################
        # STEP 2
        # Estimate Item Parameters
        # given Theta,
        #####################
        alphas, betas = _item_min_func(n_items, alphas, thetas,
                                       betas, the_sign, counts)

        if(np.abs(previous_betas - betas).max() < 1e-3):
            break
    
    return thetas, betas


def rasch_jml(dataset, discrimination=1, options=None):
    """ Estimates difficulty parameters in an IRT model
    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options
    Returns:
        difficulty: (1d array) estimates of item difficulties
    Options:
        * max_iteration: int
    """
    options = validate_estimation_options(options)

    # Defines item parameter update function
    def _item_min_func(n_items, alphas, thetas,
                       betas, the_sign, counts):
        # pylint: disable=cell-var-from-loop

        for ndx in range(n_items):
            scalar = alphas[0] * the_sign[ndx, :]

            def _beta_min(beta):
                otpt = np.exp(scalar * (thetas - beta))
                return np.log1p(otpt).dot(counts)

            # Solves for the beta parameters
            betas[ndx] = fminbound(_beta_min, -6, 6)

        return alphas, betas

    result = _jml_abstract(dataset, _item_min_func,
                           discrimination, options['max_iteration'])

    return result


class JointMLE():
    def __init__(self):
        return
    
    def fit(self, X):
        self.w, self.z = rasch_jml(X)
        return self.w