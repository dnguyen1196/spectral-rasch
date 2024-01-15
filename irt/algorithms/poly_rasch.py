import numpy as np
import scipy as sp
import pandas as pd
from scipy.special import expit as logistic_sigmoid
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm as spnorm
import time
from scipy.sparse import csr_matrix


def generate_polytomous_rasch(thetas, betas):
    """
    Generate synthetic data according to the PCM model.
    Args:
        thetas: A (n_users,) vector representing the students' abilities
        betas: A (n_tests, K) matrox representing the tests' difficulties
    Returns:
        A matrix X with shape (n_users, n_tests) generated per the PCM.
        
    """
    
    n = len(thetas)
    m, L = betas.shape # Number of items, number of levels
    results = np.zeros((n, m))
    # Compute Pnil for all (student, test, level) tuple
    # For each (student)
    for test_id, beta in enumerate(betas):
        # Go through each question to generate the students' responses.
        # Ignoring the test index for now
        
        # ability_ms_diffty[i, l] = betai - theta_l
        # This has shape (n, L)
        ability_ms_diff = thetas[:, None] - beta.reshape((1, L))
        
        # Pnk = logit(beta_n - theta_k) = prob(student n overcomes threshold k)
        p = logistic_sigmoid(ability_ms_diff)
        # Qnik = 1 - Pnik
        q = 1. - p # Complementary fail probability
        
        # Compute the running Pnik product
        # P(Xni = l) = passes all level from 1 to l, but fails from l+1 to L
        probs = np.zeros((n, L+1))
        for l in range(L+1):
            # For the two extreme cases where l = 0, or L
            if l == 0:
                probs[:, l] = np.prod(q, 1)
            elif l == L:
                probs[:, l] = np.prod(p, 1)
            else:
                # p[:, l] = [Pn1, Pn2, ... , Pnl]
                # q[:, l:] = [Qnl+1,..., QnL]
                probs[:, l] = np.prod(p[:, :l], 1) * np.prod(q[:, l:], 1)
                
        # Normalize by row to obtain grade probability
        row_sum = np.sum(probs, 1)
        # Draw student results from categorical distribution
        probs /= row_sum[:, None] # Should have shape (n, L+1)
        # Draw performance for each student individually
        for student_id in range(n):
            pi_student = probs[student_id, :]
            results[student_id, test_id] = np.random.choice(L+1, p=pi_student)
    
    return results


##############################################################################################################
#
#
#                       The spectral algorithm for the polytomous Rasch model
#
#
##############################################################################################################    
    
class SpectralAlgorithm:
    def __init__(self, lambd=0.1):
        """
        Args:
            lambd: float, regularization parameter to avoid numerical issues when computing the stationary distribution
        
        """
        self.precomputed_Y = {} # This will be used to save computations
        self.lambd = lambd
    
    def fit(self, X, sample_weight=None):
        """
        Args:
            X: Input response matrix of shape (n_users, n_itmes) 
            sample_weight: optional vector of per user sample weight
        
        
        L is the max grade level (0, 1, 2, ..., L)
        
        1. Obtain shifted estimate for each level l = 1, ..., L
            by computing the stationary distribution of the Markov chain constructed from
            Y(l-1, l)
            
            Denote these as tilde tilde theta^1, ..., tilde theta^L
        
        2. Normalize level 1 parameter to have mean 0 -> Denote as hat theta^1
        
        3. For each level l = 2,...,L, estimate the shift delta_l by
            
            sum_{j=1}^m sum_{i not j} e^{hat theta_i^{l-1}} * Y_ij(l-1,l-1)
            
            DIVIDED by
            
            sum_{j=1}^m e^{tilde theta_j^{l}} sum_{i not j} Y_{ij}(l, l-2)
            
            Recover the level l parameter estimate hat theta^l = tilde theta^l + delta_l
        
        ---------------
        Returns
            m x K parameters for the tests' difficulties
        
        """
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], ))
            
        L = int(np.max(X))
        normalized_betas = self.compute_normalized_betas(X, L, sample_weight)
        unnormalized_betas = self.compute_unnormalized_betas(X, normalized_betas, L, sample_weight)
        return unnormalized_betas
    
    def compute_normalized_betas(self, X, L, sample_weight=None):
        """Perform the first phase of the spectral algorithm to obtain level-wise normalized estimate

        Args:
            X (matrix of shape (n_users, n_items)): Input data
            L (int): Number of levels
            sample_weight (float vector, optional): User sample weight. Defaults to None.
            verbose (bool, optional): Defaults to False.

        Returns:
            Normalized betas
        """
        normalized_betas = []
        for l in range(1, L+1): # For l = 1,2,...,L
            Y_ll1 = self.construct_Y_matrix(X, l, l-1, sample_weight)
            M_ll1, d = self.construct_markov_chain(Y_ll1)
            pi = self.compute_stationary_distribution(M_ll1)
            pi = pi.flatten()
            pi = pi/d
            normalized_betas.append(np.log(pi))
        return np.array(normalized_betas)
    
    
    def compute_unnormalized_betas(self, X, normalized_betas, L, sample_weight=None):
        """

        Args:
            X: input data matrix
            normalized_betas: output of the first phase of the spectral algorithm
            L: number of levels, need to match with normalized_betas.shape[1]
            sample_weight: user-wise sample weight
        
        Returns
            The final beta estimate
        
        """
                
        unnormalized_betas = []
        shifted_betas_1 = normalized_betas[0, :] - np.mean(normalized_betas[0, :])
        unnormalized_betas.append(shifted_betas_1)
        n, m = X.shape
        
        for k in range(2, L+1):
            Y_1_km1 = self.construct_Y_matrix(X, 1, k-1, sample_weight)
            Y_0_k = self.construct_Y_matrix(X, 0, k, sample_weight)
            np.fill_diagonal(Y_1_km1, 0)
            np.fill_diagonal(Y_0_k, 0)
            
            shifted_betas_k = normalized_betas[k-1, :] - np.mean(normalized_betas[k-1,:])
            numer = Y_1_km1 * (np.exp(shifted_betas_1)).reshape((-1, 1))
            denom = Y_0_k * (np.exp(shifted_betas_k)).reshape((1, -1))
            
            delta_k = np.log(np.sum(numer)/np.sum(denom))            
            unnormalized_betas.append(shifted_betas_k + delta_k)
        
        return np.array(unnormalized_betas).T
    
        
    def construct_markov_chain(self, Y):
        """
        Given Y matrix, construct a degree-unnormalized Markov chain
        
        """
        M = np.copy(Y).astype(float)
        M = np.where(np.logical_or((M != 0), (M.T != 0)), M + self.lambd, M)
        
        m = M.shape[0]
        d = np.maximum(np.sum(M, 1), 1)
        for i in range(m):
            M[i, :] = M[i, :] / d[i]
            M[i, i] = 1. - np.sum(M[i, :])
        return M, d
    
    def compute_stationary_distribution(self, M, pi_init=None, max_iters=10000, eps=1e-6):
        """
        Compute the stationary distribution of a Markov chain
        
        """
        m = M.shape[0]
        if pi_init is None:
            pi_init = np.ones((m, )).T
        pi = pi_init
        for _ in range(max_iters):
            pi_next = (pi @ M)
            pi_next /= np.sum(pi_next)
            if np.linalg.norm(pi_next - pi) < eps:
                pi = pi_next
                break
            pi = pi_next
        return pi
    
    def construct_Y_matrix(self, X, k, k_prime, sample_weight=None):
        """Construct the Y matrix used in both phases of the spectral algorithm

        Args
            X: data
            k, k_prime: two levels needed to compute Y_{ij}^{k, k_prime}
            sample_weight: user-wise sample weight
            
        Returns
            Y matrix of shape (n_items, n_items)
        
        """
        # Note: this is the only function that directly interacts with the data X
        # It automatically handles missing data
        # assert(k <= k_prime)
        # For every pair i, j
        # Yij = #{student with score k for i and score k' for j}
        # If k' is higher -> then j is 'easier'
        # Then the Markov chain should flip this direction, since we want the parameter for j 
        # to be lower. 
        
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0], 1))
        
        if (k, k_prime) in self.precomputed_Y:
            return self.precomputed_Y[(k, k_prime)]
        is_k = csr_matrix((X == k).astype(int) * sample_weight.reshape((-1, 1)))
        is_k_prime = csr_matrix((X == k_prime).astype(int) * sample_weight.reshape((-1, 1)))
        # Yij = sum_n is_k[n, i] * is_k'[n, j]
        #     = np.sum(is_k[:, i] * is_k'[n, j])
        #     = (is_k.T @ is_k')_{ij}
        Y = (is_k.T @ is_k_prime).toarray()
        self.precomputed_Y[(k, k_prime)] = Y
        return Y


##############################################################################################################
#
#
#                       ADAPTED FROM GIRTH https://github.com/eribean/girth package
#                       MMLE and JMLE inference algorithms and helper functions
#
#
##############################################################################################################
    
from scipy.stats import norm as gaussian
from scipy.optimize import fmin_slsqp, fminbound
INVALID_RESPONSE = -99999
from scipy.special import expit
from girth.utilities import (
    condition_polytomous_response, validate_estimation_options)


def estimate_abilities_given_difficulties(X, betas, students_by_tests=True, options=None):
    """   Args:
        dataset: [n_items, n_participants] 2d array of measured responses
        betas: the estimated item difficulties
        students_by_tests: added argument to transpose intput X as the original implementation assumes a item by user matrix input
        options: dictionary with updates to default options
        
        ---
        Returns theta estimate of the student abilities
    """
    
    options = validate_estimation_options(options)
    if students_by_tests:
        dataset = X.T.astype(int)
    else:
        dataset = X.astype(int)
    
    # This function removes the bad response patterns
    cpr_result = condition_polytomous_response(dataset, trim_ends=False, _reference=0.0)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask    
    n_items, n_takers = responses.shape

    # Set initial parameter estimates to default
    thetas = np.zeros((n_takers,))

    # Initialize item parameters for iterations
    discrimination = np.ones((n_items,))
    scratch = np.zeros((n_items, betas.shape[1] + 1))

    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0

    #####################
    # Estimate theta, given betas / alpha
    # Loops over all persons
    #####################
    for ndx in range(n_takers):
        # pylint: disable=cell-var-from-loop
        response_set = responses[:, ndx]

        def _theta_min(theta, scratch):
            # Solves for ability parameters (theta)
            # Graded PCM Model
            scratch *= 0.
            scratch[:, 1:] = theta - betas
            scratch *= discrimination[:, None]
            np.cumsum(scratch, axis=1, out=scratch)
            np.exp(scratch, out=scratch)
            scratch /= np.nansum(scratch, axis=1)[:, None]

            # Probability associated with response
            values = np.take_along_axis(
                scratch, response_set[:, None], axis=1).squeeze()
            return -np.log(values[valid_response_mask[:, ndx]] + 1e-32).sum()

        thetas[ndx] = fminbound(_theta_min, -6, 6, args=(scratch,))
    return thetas


def pcm_jml(X, options=None, students_by_tests=True):
    """Estimate parameters for partial credit model.
    Estimate the discrimination and difficulty parameters for
    the partial credit model using joint maximum likelihood.
    
    Args:
        X: [n_participants, n_tests] 2d array of measured responses
        students_by_tests: added argument to transpose intput X as the original implementation assumes a item by user matrix input
        options: dictionary with updates to default options
    Returns:
        difficulty: (2d array) estimates of item difficulties x item thresholds
    Options:
        * max_iteration: int
    """
    dataset = X.T.astype(int) if students_by_tests else X.astype(int)
    options = validate_estimation_options(options)

    cpr_result = condition_polytomous_response(dataset, _reference=0.0)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask    
    n_items, n_takers = responses.shape

    # Set initial parameter estimates to default
    thetas = np.zeros((n_takers,))

    # Initialize item parameters for iterations
    discrimination = np.ones((n_items,))
    betas = np.full((n_items, item_counts.max() - 1), np.nan)
    scratch = np.zeros((n_items, betas.shape[1] + 1))

    for ndx in range(n_items):
        item_length = item_counts[ndx] - 1
        betas[ndx, :item_length] = np.linspace(-1, 1, item_length)

    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0        

    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()
        previous_betas = betas.copy()

        #####################
        # STEP 1
        # Estimate theta, given betas / alpha
        # Loops over all persons
        #####################
        for ndx in range(n_takers):
            # pylint: disable=cell-var-from-loop
            response_set = responses[:, ndx]

            def _theta_min(theta, scratch):
                # Solves for ability parameters (theta)

                # Graded PCM Model
                scratch *= 0.
                scratch[:, 1:] = theta - betas
                scratch *= discrimination[:, None]
                np.cumsum(scratch, axis=1, out=scratch)
                np.exp(scratch, out=scratch)
                scratch /= np.nansum(scratch, axis=1)[:, None]

                # Probability associated with response
                values = np.take_along_axis(
                    scratch, response_set[:, None], axis=1).squeeze()
                return -np.log(values[valid_response_mask[:, ndx]] + 1e-313).sum()

            thetas[ndx] = fminbound(_theta_min, -6, 6, args=(scratch,))

        # Recenter theta to identify model
        thetas -= thetas.mean()
        thetas /= thetas.std(ddof=1)

        #####################
        # STEP 2
        # Estimate Betas / alpha, given Theta
        # Loops over all items
        #####################
        for ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            # Compute ML for static items
            response_set = responses[ndx]

            def _alpha_beta_min(estimates):
                # PCM_Model
                kernel = thetas[:, None] - estimates[None, :]
                kernel *= estimates[0]
                kernel[:, 0] = 0
                np.cumsum(kernel, axis=1, out=kernel)
                np.exp(kernel, out=kernel)
                kernel /= np.nansum(kernel, axis=1)[:, None]
                # Probability associated with response
                values = np.take_along_axis(
                    kernel, response_set[:, None], axis=1).squeeze()
                return -np.log(values[valid_response_mask[ndx]]).sum()

            # Solves jointly for parameters using numerical derivatives
            initial_guess = np.concatenate(([discrimination[ndx]],
                                            betas[ndx, :item_counts[ndx]-1]))
            otpt = fmin_slsqp(_alpha_beta_min, initial_guess,
                              disp=False,
                              bounds=[(.25, 4)] + [(-6, 6)] * (item_counts[ndx]-1))

            discrimination[ndx] = 1.
            betas[ndx, :item_counts[ndx]-1] = otpt[1:]

        # Check termination criterion
        if(np.abs(betas - previous_betas).max() < 1e-3):
            break

    shift = np.mean(betas[:, 0])
    betas = betas - shift
    return betas
    

from girth.utilities.latent_ability_distribution import LatentPDF
from girth.unidimensional.polytomous.partial_integrals_poly import _credit_partial_integral
from girth.unidimensional.polytomous.ability_estimation_poly import _ability_eap_abstract


def pcm_mml(X, options=None, students_by_tests=True):
    """Estimate parameters for partial credit model.
    Estimate the discrimination and difficulty parameters for
    the partial credit model using marginal maximum likelihood.
    Args:
        X: [n_participants, n_tests] 2d array of measured responses        
        students_by_tests: added argument to transpose intput X as the original implementation assumes a item by user matrix input
        options: dictionary with updates to default options
    Returns:
        difficulty: (2d array) estimates of item difficulties x item thresholds
    Options:
        * estimate_distribution: Boolean    
        * number_of_samples: int >= 5       
        * max_iteration: int
        * distribution: callable
        * quadrature_bounds: (float, float)
        * quadrature_n: int
    """
    dataset = X.T.astype(int) if students_by_tests else X.astype(int)
    options = validate_estimation_options(options)

    cpr_result =  condition_polytomous_response(dataset, trim_ends=False, _reference=0.0)
    responses, item_counts, valid_response_mask = cpr_result
    invalid_response_mask = ~valid_response_mask

    n_items = responses.shape[0]

    # Quadrature Locations
    latent_pdf = LatentPDF(options)
    theta = latent_pdf.quadrature_locations

    # Initialize difficulty parameters for estimation
    betas = np.full((n_items, item_counts.max()), np.nan)
    discrimination = np.ones((n_items,))
    partial_int = np.ones((responses.shape[1], theta.size))

    # Not all items need to have the same
    # number of response categories
    betas[:, 0] = 0
    for ndx in range(n_items):
        betas[ndx, 1:item_counts[ndx]] = np.linspace(-1, 1, item_counts[ndx]-1)

    # Set invalid index to zero, this allows minimal
    # changes for invalid data and it is corrected
    # during integration
    responses[invalid_response_mask] = 0

    #############
    # 1. Start the iteration loop
    # 2. Estimate Dicriminatin/Difficulty Jointly
    # 3. Integrate of theta
    # 4. minimize and repeat
    #############
    for iteration in range(options['max_iteration']):
        previous_discrimination = discrimination.copy()
        previous_betas = betas.copy()

        # Quadrature evaluation for values that do not change
        # This is done during the outer loop to address rounding errors
        # and for speed
        partial_int = np.ones((responses.shape[1], theta.size))
        for item_ndx in range(n_items):
            partial_int *= _credit_partial_integral(theta, betas[item_ndx],
                                                    discrimination[item_ndx],
                                                    responses[item_ndx],
                                                    invalid_response_mask[item_ndx])
        # Estimate the distribution if requested
        distribution_x_weight = latent_pdf(partial_int, iteration)
        partial_int *= distribution_x_weight        

        # Loop over each item and solve for the alpha / beta parameters
        for item_ndx in range(n_items):
            # pylint: disable=cell-var-from-loop
            item_length = item_counts[item_ndx]
            new_betas = np.zeros((item_length))

            # Remove the previous output
            old_values = _credit_partial_integral(theta, previous_betas[item_ndx],
                                                  previous_discrimination[item_ndx],
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])
            partial_int /= old_values

            def _local_min_func(estimate):
                new_betas[1:] = estimate[1:]
                new_values = _credit_partial_integral(theta, new_betas,
                                                      estimate[0],
                                                      responses[item_ndx],
                                                      invalid_response_mask[item_ndx])
                new_values *= partial_int
                otpt = np.sum(new_values, axis=1)
                return -np.log(otpt).sum()

            # Initial Guess of Item Parameters
            initial_guess = np.concatenate(([discrimination[item_ndx]],
                                            betas[item_ndx, 1:item_length]))

            otpt = fmin_slsqp(_local_min_func, initial_guess,
                              disp=False,
                              bounds=[(.25, 4)] + [(-6, 6)] * (item_length - 1))

            discrimination[item_ndx] = 1.
            betas[item_ndx, 1:item_length] = otpt[1:]

            new_values = _credit_partial_integral(theta, betas[item_ndx],
                                                  discrimination[item_ndx],
                                                  responses[item_ndx],
                                                  invalid_response_mask[item_ndx])

            partial_int *= new_values

        if np.abs(betas - previous_betas).max() < 1e-3:
            break

    output_betas = betas[:,1:]
    shift = np.mean(output_betas[:, 0])
    output_betas = output_betas - shift
    return output_betas