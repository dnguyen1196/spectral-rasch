from scipy.optimize import fminbound, minimize, LinearConstraint
import numpy as np



__all__ = ["rasch_conditional"]

INVALID_RESPONSE = -99999

def _symmetric_functions(betas):
    """Computes the symmetric functions based on the betas
        Indexes by score, left to right
    """
    polynomials = np.c_[np.ones_like(betas), np.exp(-betas)]

    # This is an easy way to compute all the values at once,
    # not necessarily the fastest
    otpt = 1
    for polynomial in polynomials:
        otpt = np.convolve(otpt, polynomial)
    return otpt

def trim_response_set_and_counts(response_sets, counts):
    """ Trims all true or all false responses from the response set/counts.
    Args:
        response_set:  (2D array) response set by persons obtained by running
                        numpy.unique
        counts:  counts associated with response set
    Returns:
        response_set: updated response set with removal of undesired response patterns
        counts: updated counts to account for removal
    """
    # Find any missing data
    bad_mask = response_sets == INVALID_RESPONSE

    # Remove response sets where output is all true/false
    mask = ~(np.ma.var(np.ma.masked_array(response_sets, bad_mask), axis=0) == 0)
    response_sets = np.ma.masked_array(response_sets, bad_mask)[:, mask]
    counts = counts[mask]
    return response_sets, counts


def _elementary_symmetric_polynomial(sub_betas):
    """Computes the symmetric functions based on the betas
        Indexes by score, left to right
    """
    polynomials = np.c_[np.ones_like(sub_betas), np.exp(-sub_betas)]
    # This is an easy way to compute all the values at once,
    # not necessarily the fastest
    otpt = 1
    for polynomial in polynomials:
        otpt = np.convolve(otpt, polynomial)
    return otpt


def _symmetric_functions_sparse(betas, response_indices, unique_response_patterns, num_positive_responses):
    # For each unique response pattern, evaluate the symmetric function at betas
    _, num_unique_patterns = unique_response_patterns.shape 
    gamma_R = np.zeros(num_unique_patterns)

    for r in range(num_unique_patterns):
        sub_betas = betas[response_indices[r]]
        gamma_R[r] = _elementary_symmetric_polynomial(sub_betas)[num_positive_responses[r]]

    return gamma_R


def rasch_conditional_modified(dataset, discrimination=1, max_iters=1000, return_beta=True, verbose=False):
    """ Estimates the difficulty parameters in a Rasch IRT model
    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options
    Returns:
        difficulty: (1d array) estimates of item difficulties
    Options:
        * max_iteration: int
    Notes:
        This function sets the sum of difficulty parameters to 
        zero for identification purposes
    """
    n_items = dataset.shape[0]
    unique_response_patterns, response_patterns_counts = np.unique(dataset, axis=1, return_counts=True)

    betas = np.zeros((n_items, ))
    identifying_mean = 0.0

    # Remove the zero and full count values
    unique_response_patterns, response_patterns_counts = trim_response_set_and_counts(unique_response_patterns, response_patterns_counts)
    unique_response_patterns = np.ma.masked_equal(unique_response_patterns, INVALID_RESPONSE, copy=False)
    num_positive_responses = np.ma.sum(unique_response_patterns, axis=0) # Number of positive responses per response pattern (more than one user may responds this way)


    num_total_responses = np.ma.count(unique_response_patterns, 0)
    good_cols = np.where(num_total_responses != 0)[0]

    unique_response_patterns = unique_response_patterns[:, good_cols]
    num_positive_responses = num_positive_responses[good_cols]
    num_total_responses = num_total_responses[good_cols]
    response_patterns_counts = response_patterns_counts[good_cols]
    response_indices = [
        np.where(unique_response_patterns[:, r] != INVALID_RESPONSE)[0] for r in range(len(response_patterns_counts))
    ]

    if verbose:
        print("unique response patterns", unique_response_patterns)
        print("num_positive_responses", num_positive_responses)
        print("num total responses", np.ma.count(unique_response_patterns, 0))    

    for _ in range(max_iters):
        previous_betas = betas.copy()

        for ndx in range(n_items): 
            # Do coordinate ascent

            def min_func(estimate):
                betas[ndx] = estimate
                gamma = _symmetric_functions_sparse(betas, response_indices, unique_response_patterns, num_positive_responses)
                denom =  np.log(gamma).dot(response_patterns_counts)
                numer = np.ma.sum(unique_response_patterns * betas[:, None], axis=0).dot(response_patterns_counts) # The numerator is generally correct
                return numer + denom

            # Solve for the difficulty parameter
            betas[ndx] = fminbound(min_func, -10, 10)

            # recenter
            betas += (identifying_mean - betas.mean())

        # Check termination criterion
        if np.abs(betas - previous_betas).max() < 1e-3:
            break

    difficulty = betas / discrimination
    
    if return_beta:
        difficulty -= np.mean(difficulty)
        return difficulty
    
    else:
        z = np.exp(difficulty)
        z /= np.sum(z)
        return z


def rasch_conditional_gradient(dataset, discrimination=1, max_iters=1000, return_beta=True, verbose=False):
    """ Estimates the difficulty parameters in a Rasch IRT model
    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options
    Returns:
        difficulty: (1d array) estimates of item difficulties
    Options:
        * max_iteration: int
    Notes:
        This function sets the sum of difficulty parameters to 
        zero for identification purposes
    """
    n_items = dataset.shape[0]
    unique_response_patterns, response_patterns_counts = np.unique(dataset, axis=1, return_counts=True)

    betas = np.zeros((n_items, ))

    # Remove the zero and full count values
    unique_response_patterns, response_patterns_counts = trim_response_set_and_counts(unique_response_patterns, response_patterns_counts)
    unique_response_patterns = np.ma.masked_equal(unique_response_patterns, INVALID_RESPONSE, copy=False)
    num_positive_responses = np.ma.sum(unique_response_patterns, axis=0) # Number of positive responses per response pattern (more than one user may responds this way)

    num_total_responses = np.ma.count(unique_response_patterns, 0)
    good_cols = np.where(num_total_responses != 0)[0]

    unique_response_patterns = unique_response_patterns[:, good_cols]
    num_positive_responses = num_positive_responses[good_cols]
    num_total_responses = num_total_responses[good_cols]
    response_patterns_counts = response_patterns_counts[good_cols]
    response_indices = [
        np.where(unique_response_patterns[:, r] != INVALID_RESPONSE)[0] for r in range(len(response_patterns_counts))
    ]

    if verbose:
        print("unique response patterns", unique_response_patterns)
        print("num_positive_responses", num_positive_responses)
        print("num total responses", np.ma.count(unique_response_patterns, 0))    


    def log_lik(betas):
        gamma = _symmetric_functions_sparse(betas, response_indices, unique_response_patterns, num_positive_responses)
        denom =  np.log(gamma).dot(response_patterns_counts)
        numer = np.ma.sum(unique_response_patterns * betas[:, None], axis=0).dot(response_patterns_counts) # The numerator is generally correct
        return numer + denom

    # def log_lik(estimate):
    #     partial_conv = _symmetric_functions(estimate)
    #     full_convolution = np.convolve([1, np.exp(-estimate)], partial_conv)
    #     denominator = full_convolution[num_positive_responses]
    #     return (np.sum(unique_response_patterns * betas[:,None], axis=0).dot(response_patterns_counts) + 
    #             np.log(denominator).dot(response_patterns_counts))

    betas0 = np.zeros((n_items, ))
    constraint = LinearConstraint(np.ones((1, n_items)), 0, 0)
    res = minimize(log_lik, betas0, constraints=[constraint])
    betas = res["x"]
    return betas


def rasch_conditional_original(dataset, discrimination=1, max_iters=10000, return_beta=True, verbose=False):
    """ Estimates the difficulty parameters in a Rasch IRT model
    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options
    Returns:
        difficulty: (1d array) estimates of item difficulties
    Options:
        * max_iteration: int
    Notes:
        This function sets the sum of difficulty parameters to 
        zero for identification purposes
    """
    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    
    # Initialize all the difficulty parameters to zeros
    # Set an identifying_mean to zero
    ##TODO: Add option to specifiy position
    betas = np.zeros((n_items, ))
    identifying_mean = 0.0

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)
    response_set_sums = unique_sets.sum(axis=0)

    for _ in range(max_iters):
        previous_betas = betas.copy()

        for ndx in range(n_items):
            partial_conv = _symmetric_functions(np.delete(betas, ndx))

            def min_func(betai):
                betas[ndx] = betai
                full_convolution = np.convolve([1, np.exp(-betai)], partial_conv)
                denominator = full_convolution[response_set_sums]
                return (np.sum(unique_sets * betas[:,None], axis=0).dot(counts) + 
                        np.log(denominator).dot(counts))

            # Solve for the difficulty parameter
            betas[ndx] = fminbound(min_func, -5, 5)

            # recenter
            betas += (identifying_mean - betas.mean())

        # Check termination criterion
        if np.abs(betas - previous_betas).max() < 1e-3:
            break

    difficulty = betas / discrimination
    
    if return_beta:
        difficulty -= np.mean(difficulty)
        return difficulty
    
    else:
        z = np.exp(difficulty)
        z /= np.sum(z)
        return z


rasch_conditional = rasch_conditional_original