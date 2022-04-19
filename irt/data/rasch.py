import numpy as np
INVALID_RESPONSE = -99999


def generate_data(betas, thetas, p, missing_data_value=INVALID_RESPONSE):
    # For each student
    m = len(betas)
    n = len(thetas)
    performances = np.zeros((m, n), dtype=np.int)
    
    for i in range(m):
        betai = betas[i]
        for l in range(n):
            thetal = thetas[l]
            if np.random.rand() < p:
                if np.random.rand() < 1./(1 + np.exp(-(thetal - betai))):
                    # If the students solve the problem
                    performances[i, l] = 1
                else:
                    performances[i, l] = 0
            else:
                performances[i, l] = missing_data_value
        
    return performances
