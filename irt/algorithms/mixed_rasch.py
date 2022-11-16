import numpy as np
import scipy.sparse as sparse_sp
from sklearn.cluster import KMeans
from .spectral_estimator import spectral_joint_estimate, spectral_estimate


INVALID_RESPONSE = -99999

# TODO: incorporate kernel smoothing for the mixed Rasch estimator


class MixedRaschEstimator:
    def __init__(self, K, n, m):
        self.K = K
        self.n = n # Number of users
        self.m = m # Number of items
        
    def fit(self, A, max_iters=100):
        
        
        return
    
    def e_step(self, A, theta, beta):
        
        
        return
    
    def m_step(self, A, qz, theta, beta):
        
        return
    
    def spectral_init(self, A, K):
        # Run spectral clustering on the columns of A 
        A_sparse = np.where(A != INVALID_RESPONSE, A*2-1, 0)
        A_sparse = sparse_sp.csr_matrix(A_sparse)
        U, s, Vt = sparse_sp.linalg.svds(A_sparse, self.K)
        
        # Low rank approximation of A_lr
        A_lr = (np.diag(s) @ Vt).T
        kmeans = KMeans(self.K)
        kmeans.fit(A_lr)
        labels, centers = kmeans.labels_, kmeans.cluster_centers_ # The centers can also be used to obtain estimate
        
        betas = []
        thetas = []
        
        for k in range(self.K):
            A_subk = np.array([A[:, i] for i in range(len(A)) if labels[i] == k])
            betak = spectral_joint_estimate(np.transpose(A_subk))
            betas.append(betak)
            # thetas.append(thetak)
        
        return betas, thetas
    
    

