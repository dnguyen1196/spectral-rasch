import numpy as np
from scipy.sparse import csc_matrix
import gpytorch as gp
import torch as th
import time

INVALID_RESPONSE = -99999


def construct_markov_chain(A, lambd=0.1):
    m, n = A.shape
    D = np.ma.masked_equal(A, INVALID_RESPONSE, copy=False)
    
    D_compl = 1. - D
    M = np.ma.dot(D, D_compl.T)
    np.fill_diagonal(M, 0)
    M = np.round(M)
    
    # Add regularization to the 'missing' entries
    M = np.where(np.logical_or((M != 0), (M.T != 0)), M+lambd, M)
    
    d = []
    for i in range(m):
        di = max(np.sum(M[i, :]), 1)
        d.append(di)
        M[i, :] /= max(d[i], 1)
        M[i, i] = 1. - np.sum(M[i, :])

    d = np.array(d)
    return M, d


def construct_regularized_markov_chain(A, kernel_function, X, Y):
    
    
    
    return
    

def spectral_estimate(A, max_iters=10000, return_beta=True, lambd=1, eps=1e-6):
    """Estimate the hidden parameters according to the Rasch model, either for the tests' difficulties
    or the students' abilities. Following the convention of Girth https://eribean.github.io/girth/docs/quickstart/quickstart/
    the rows correspond to the problems and the columns correspond to the students.
    """

    M, d = construct_markov_chain(A, lambd=lambd)
    M = csc_matrix(M)
    
    m = len(A)        
    pi = np.ones((m,)).T
    for _ in range(max_iters):
        pi_next = (pi @ M)
        pi_next /= np.sum(pi_next)
        if np.linalg.norm(pi_next - pi) < eps:
            pi = pi_next
            break
        pi = pi_next
        
    pi = pi.T/d
    if return_beta:
        beta = np.log(pi)
        beta = beta - np.mean(beta)
        return beta
    
    return pi/np.sum(pi)


th.set_default_dtype(th.float32)


class RegularizedSpectral:
    def __init__(self):
        return
    
    def fit(self, A, kernel_function=None):
        # Perform kernel smoothing on A
        
        if kernel_function is None:
            kernel_function = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
        
        
        return
    
    def construct_regularized_markov_chain(self, A, X, Y, kernel_function):
        assert(Y is not None)
        if X is None:
            A_smoothed = self.kernel_smooth_without_user_features(A, Y, kernel_function)
        else:
            A_smoothed = self.kernel_smooth_with_user_features(A, X, Y, kernel_function)
        
        # A_smoothed will no longer be a binary matrix, will its entries still be between 0 and 1?
        
        return
    
    def kernel_smooth_with_user_features(self, A, X, Y, kernel, large_A = False, verbose= True, stride = 1):
        m, n = A.shape
        A_replaced = np.where(A == 0, -1, A)
        A_replaced = np.where(A_replaced == INVALID_RESPONSE, 0, A_replaced)
        A_replaced = A_replaced.flatten()
        mask = np.where(A_replaced != 0, 1, 0)

        # Convolve X and Y
        XY_features = []
        for i in range(len(Y)):
            for l in range(len(X)):
                yi = np.array(np.atleast_2d(Y[i, :]))
                xl = np.array(np.atleast_2d(X[l, :]))
                xy = np.concatenate([yi, xl], 1).T
                XY_features.append(xy.squeeze())
        
        XY_features = th.tensor(XY_features)
        
        # Break the computation into chunks to make it less memory intensive
        if large_A:
            if verbose:
                print(f"Large matrix A, smoothen each row one by one ")
            # Convert mask into a matrix for easy multiplication
            # mask = np.array([mask for _ in range(len(X))])
            
            A_smoothed = []
            for i in range(0, len(Y), stride):
                start = time.time()
                XYi_features = XY_features[i * len(X): min((i+stride) * len(X), len(XY_features)), :]
                extract_time = time.time()
                k_XYi_XY = kernel(XYi_features, XY_features).detach().numpy()
                kernel_time = time.time()
                k_XYi_XY = k_XYi_XY * mask
                masking_time = time.time()
                k_XYi_XY = k_XYi_XY / k_XYi_XY.sum(1)[:, np.newaxis]
                normalizing_time = time.time()
                Ai_smoothed = k_XYi_XY @ A_replaced
                mult_time = time.time()
                if verbose and stride == 1:
                    print(f"Filling row {i}: extracting took {extract_time - start}, kernelling took {kernel_time - extract_time}, masking took {masking_time - kernel_time}, normalizing took {normalizing_time - masking_time}, multiplying took {mult_time - normalizing_time}")
                # elif verbose and stride != 1:
                    # print(f"Filling rows {i} to {i+stride}: extracting took {extract_time - start}, kernelling took {kernel_time - extract_time}, masking took {masking_time - kernel_time}, normalizing took {normalizing_time - masking_time}, multiplying took {mult_time - normalizing_time}")
                
                # if stride == 1:
                A_smoothed.append(Ai_smoothed)
                # else:
                #     Ai_smoothed
                    
            A_smoothed = np.array(A_smoothed)
            assert(A_smoothed.shape == (m, n))
            
        else:
            kXY_XY = kernel(XY_features).detach().numpy()
            kXY_XY = kXY_XY * mask
            kXY_XY = kXY_XY / kXY_XY.sum(1)[:, np.newaxis]
            A_smoothed = kXY_XY @ A_replaced
            A_smoothed = A_smoothed.reshape((m, n))
            
        return (1 + A_smoothed)/2
    
    def kernel_smooth_without_user_features(self, A, Y, kernel_function):
        A_replaced = np.where(A == 0, -1, A)
        A_replaced = np.where(A_replaced == INVALID_RESPONSE, 0, A_replaced)

        m, n = A.shape
        kYY = kernel_function(th.tensor(Y)).detach().numpy() # This should have shape (m, m)
        assert(kYY.shape == (m, m))

        A_smoothed = []
        for l in range(n):
            # The proper way of doing this is to not assign weight to the columns corresponding to the missing entires
            mask = np.where(A_replaced[:, l] != 0, 1, 0)
            kYY_masked = kYY * mask
            kYY_masked = kYY_masked / kYY_masked.sum(1)[:, np.newaxis]
            A_smoothed.append(kYY_masked @ A_replaced[:, l])
            
        A_smoothed = np.array(A_smoothed)
        A_smoothed = (A_smoothed + 1)/2
        return A_smoothed.T
    

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, mean=None, kernel=None, var_dist=None):
        if var_dist is None or var_dist == "MeanField":
            variational_distribution = gp.variational.MeanFieldVariationalDistribution(inducing_points.size(0))
        if var_dist == "Cholesky":
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        
        if mean == "Constant" or mean is None:
            self.mean_module = gp.means.ConstantMean()
        if mean == "Linear":
            self.mean_module = gp.means.LinearMean(inducing_points.size(1))
        if mean == "Zero":
            self.mean_module = gp.means.ZeroMean()
        
        if kernel == "RBF":
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
        if kernel == "Matern" or kernel is None:
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel())
        self.float()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def get_responses(A, X, Y):
    m, n = A.shape
    
    # Convolve X and Y
    XY_features = []
    responses = []
    
    for i in range(m):
        for l in range(n):
            if A[i, l] != INVALID_RESPONSE:
                yi = np.array(np.atleast_2d(Y[i, :]))
                xl = np.array(np.atleast_2d(X[l, :]))
                xy = np.concatenate([yi, xl], 1).T.squeeze()
                XY_features.append(xy)
                responses.append(A[i, l])
    
    return th.tensor(XY_features, dtype=th.float32), th.tensor(responses).float()


class KernelSmoother():
    def __init__(self, kernel_function=None, mean_function=None, var_dist=None,):
        if kernel_function is None:
            kernel_function = "Matern"
        if mean_function is None:
            mean_function = "Constant"
        if var_dist is None:
            var_dist = "MeanField"
        
        self.kernel_function = kernel_function
        self.mean_function = mean_function
        self.var_dist = var_dist
    
    def fit(self, A, X, Y, n_inducing_points=None, lr=0.01, max_epochs=100, minibatch=None, verbose=False, A_true=None, eps=1e-3, report=10):
        Z_val = None
        responses_val = None
        if A_true is not None:
            Z_val, responses_val = get_responses(A_true, X, Y)
            
        Z, responses = get_responses(A, X, Y)
        
        N = len(Z)
        if n_inducing_points is None:
            n_inducing_points = N
        random_indices = np.random.choice(N, size=(n_inducing_points,), replace=True)
        inducing_points = th.tensor(Z[random_indices, :], dtype=th.float32)
        
        self.approximate_gp = GPModel(inducing_points, mean=self.mean_function, kernel=self.kernel_function, var_dist=self.var_dist)        
        
        
        # Assuming that Z has been converted to torch tensor
        assert(len(Z) == len(responses))
        
        likelihood = gp.likelihoods.BernoulliLikelihood()
        self.likelihood = likelihood

        elbo = gp.mlls.VariationalELBO(likelihood, self.approximate_gp, len(responses))
        
        optimizer = th.optim.Adam([
            {'params': self.approximate_gp.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr)
        
        if minibatch is None:
            minibatch = len(responses)
        
        train_dataset = TensorDataset(Z, responses)
        train_loader = DataLoader(train_dataset, batch_size=minibatch)
        
        cur_epoch_loss = np.inf

        start = time.time()
        
        for it in range(max_epochs):
            epoch_loss = 0.
            for batch_Z, batch_responses in train_loader:
                optimizer.zero_grad()
                output = self.approximate_gp(batch_Z)
                loss = -elbo(output, batch_responses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().numpy()
            
            if verbose and it % report == 0:
                err_estimate = "N/A"
                if Z_val is not None:
                    responses_pred = self.predict_prob(Z_val, self.likelihood)
                    err_estimate = th.sum(th.square(responses_pred - responses_val))
                print(f"Epoch {it}: loss = {epoch_loss}, err = {err_estimate}, time-elapsed {time.time()- start}")
                
            if np.abs(epoch_loss - cur_epoch_loss) < eps:
                break
            
        if verbose:
            err_estimate = "N/A"
            if Z_val is not None:
                responses_pred = self.predict_prob(Z_val, self.likelihood)
                err_estimate = th.sum(th.square(responses_pred - responses_val))
            print(f"Epoch {it}: loss = {epoch_loss}, err = {err_estimate}")

    
    def predict_prob(self, Z, likelihood=None):
        if likelihood is None:
            likelihood = self.likelihood
        output = self.approximate_gp(Z)
        p = likelihood(output).probs.detach()
        return p
    
    def sample_posterior_prob(self, Z, likelihood=None, n_samples=100):
        if likelihood is None:
            likelihood = self.likelihood
        
        post_dist = self.approximate_gp(Z)
        f_sampled = post_dist.sample_n(n_samples)
        p_sampled = likelihood(f_sampled).probs.detach().numpy()
        return p_sampled
    

class MultiLayerNeuralNetwork(th.nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes):
        super(MultiLayerNeuralNetwork, self).__init__()
        self.layers = th.nn.ModuleList()
        hidden_layer_sizes = [input_dim] + hidden_layer_sizes # Last layer has dimension 1
        
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(
                th.nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i])
            )
            self.layers.append(
                th.nn.Sigmoid()
            )
        self.layers.append(
            th.nn.Linear(hidden_layer_sizes[-1], 1)
        )

    def forward(self, X): 
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

class LogisticRegression(th.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = th.nn.Linear(input_dim, 1)
    def forward(self, x):
        outputs = th.sigmoid(self.linear(x))
        return outputs


class NeuralSmoother():
    def __init__(self, nnet):
        self.nnet = nnet
    
    def fit(self, A, X, Y, lr=0.01, max_epochs=1000, minibatch=None, verbose=False, A_true=None, eps=1e-3, report=10):
        Z_val = None
        responses_val = None
        if A_true is not None:
            Z_val, responses_val = get_responses(A_true, X, Y)
            
        Z, responses = get_responses(A, X, Y)
        # responses = responses.expand(-1, 1)
        
        optimizer = th.optim.Adam([
            {'params': self.nnet.parameters()},
        ], lr=lr, weight_decay=0.001)
        
        if minibatch is None:
            minibatch = len(responses)
        
        train_dataset = TensorDataset(Z, responses)
        train_loader = DataLoader(train_dataset, batch_size=minibatch)
        
        cur_epoch_loss = np.inf
        start = time.time()
        
        criterion = th.nn.BCEWithLogitsLoss()
        
        for it in range(max_epochs):
            epoch_loss = 0.
            for batch_Z, batch_responses in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.nnet(batch_Z).flatten(), batch_responses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().numpy()
            
            if verbose and it % report == 0:
                err_estimate = "N/A"
                if Z_val is not None:
                    responses_pred = self.predict_prob(Z_val)
                    err_estimate = th.sum(th.square(responses_pred - responses_val))
                print(f"Epoch {it}: loss = {epoch_loss}, err = {err_estimate}, time-elapsed {time.time()- start}")
                
            if np.abs(epoch_loss - cur_epoch_loss) < eps:
                break
            
        if verbose:
            err_estimate = "N/A"
            if Z_val is not None:
                responses_pred = self.predict_prob(Z_val)
                err_estimate = th.sum(th.square(responses_pred - responses_val))
            print(f"Epoch {it}: loss = {epoch_loss}, err = {err_estimate}")
    
    def predict_prob(self, Z):
        return th.sigmoid(self.nnet(Z)).detach().flatten()