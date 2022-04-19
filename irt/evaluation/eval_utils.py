import numpy as np
from sklearn.metrics import roc_auc_score
from girth import ability_mle, ability_map, ability_eap


INVALID_RESPONSE = -99999


def graded_to_binary(A, f):
    for i in range(len(A)):
        for j in range(A.shape[1]):
            A[i, j] = f(A[i, j]) if A[i, j] != INVALID_RESPONSE else INVALID_RESPONSE
    
    return A


def partition_data(A, p=0.8, large_m=False):
    m, n = A.shape[0], A.shape[1]
    A_train = np.copy(A)
    test_data = []
    
    if not large_m:
        # For each student, removes a portion of the data
        for j in range(n):
            responses = A[:, j]
            responses_idx = np.where(responses != INVALID_RESPONSE)[0]
            num_response = len(responses_idx)
            subset_train = np.random.choice(
                num_response, size=(max(int(p * num_response), 1),), replace=False
            )
            subset_test = np.delete(responses_idx, subset_train)
            
            A_train[subset_test, j] = INVALID_RESPONSE
            for i in subset_test:
                test_data.append(((i, j), responses[i]))
    
    else: # If we have more m than n
        for i in range(m):
            responses = A[i, :]
            responses_idx = np.where(responses != INVALID_RESPONSE)[0]
            num_response = len(responses_idx)
            subset_train = np.random.choice(
                num_response, size=(max(int(p * num_response), 1),), replace=False
            )
            subset_test = np.delete(responses_idx, subset_train)
            A_train[i, subset_test] = INVALID_RESPONSE
            for j in subset_test:
                test_data.append(((i, j), responses[j]))
    
    return A_train, test_data


def partition_data_sparse(test_student_response, p=0.8, seed=None):
    # For very large dataset
    
    return


def evaluate_auc(model, A_train, test_data, verbose=False, ignore_nan=True):
    # Run the model on the training data, 
    z_est = model(A_train)
    if verbose:
        print("z_estimate", z_est)
        
    z_est = z_est/ np.sum(z_est)
    difficulties = np.exp(z_est)
    
    # Use Girth's built in functionality to estimate the abilities
    abilities = ability_mle(A_train, difficulties, 1)
    if verbose:
        print("abilities", abilities)
    # Replace nan by average abilities
    # if ignore_nan:
    y_true = []
    y_score = []
    
    for (test_id, student_id), y in test_data:
        y_true.append(y)
        y_score.append(1./(1 + np.exp(-(abilities[student_id] - difficulties[test_id]))))
        
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    non_nanindex = np.where(np.logical_not(np.isnan(y_true)) & np.logical_not(np.isnan(y_score)))[0]
    y_true = y_true[non_nanindex]
    y_score = y_score[non_nanindex]
    
    if verbose:
        print("y_score", y_score)
        print("y_true", y_true)
        
    return roc_auc_score(y_true, y_score)


def evaluate_model(A, model, p=0.8, n_trials=100, seed=None):
    np.random.seed(seed)
    avg_auc = 0.
    for _ in range(n_trials):
        A_train, test_data = partition_data(A, p)
        avg_auc += 1./n_trials * evaluate_auc(model, A_train, test_data)
    return avg_auc


def cramer_rao_lower_bound(beta_tests, theta_students, p):
    # The log likelihood should factor into individual tests
    m = len(beta_tests)
    n = len(theta_students)
    var = []
    for i in range(m):
        I_betai = 0.
        betai = beta_tests[i]

        for l in range(n):
            thetal = theta_students[l]
            I_betai += p * np.exp(-(thetal-betai))/(1+np.exp(-(thetal- betai)))**2
        var.append(1./I_betai)
    return np.array(var)