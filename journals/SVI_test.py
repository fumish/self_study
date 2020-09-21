# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Experiment for Stochastic Variational Inference
# + In this notebook, we consider about stochastic variational inference, published by Hoffman+ 2013.
# + In the journal, Latent-Dirichlet Allocation(LDA) is formulated by the inference.
#     + However, connection between the update rule by the exponential family and the the one by the LDA example is slight difficult
#     + Thus, first consider about the most simple model GMM here in order to understand the performance the inference.

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
sys.path.append("C:\\Users\\user\\Documents\\GitHub\\LearningModels\\lib")

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi, gammaln, logsumexp, softmax

from util import GaussianMixtureModel
# -

# data settings
n = 100000
M = 10 # data dimenension
data_seed = 20200920
np.random.seed(data_seed)

# data generation
K0 = 8
true_ratio = np.random.dirichlet([1]*K0)
true_mean = np.random.normal(size = (K0, M), scale = 2)

train_X, _, data_label = GaussianMixtureModel.rvs(true_ratio, true_mean, precision = np.ones((K0, M)), size = n)

for k in range(K0):
    ind = np.where(data_label == k)[0]
    plt.scatter(train_X[ind,0], train_X[ind,1])
plt.show()


# ## Preliminary for learning
# + For every experiment, we utilize an e-step and derivative of the objective function.
# + So they are prepared here.

# +
def _e_step(X, alpha, beta, gamma):
    """
    e step update
    """
    
    n, M = X.shape
    K = len(alpha)
    
    # update for local parameters
    est_phi = np.array([
        psi(alpha) - psi(alpha.sum()) 
        -0.5*((X[i,:].repeat(K).reshape(M,K).T - gamma / beta[:,np.newaxis])**2).sum(axis = 1) 
        - M/2/beta -M/2*np.log(2*np.pi)
    for i in range(n)])
    est_label = softmax(est_phi, axis = 1)
    
    return est_phi, est_label

def _calc_derivative(X, label, alpha, beta, gamma, 
                     pri_alpha, pri_beta, total_n):
    """
    derivative of the objective function
    """
    n_X = X.shape[0]
    sample_ratio = total_n / n_X
    
    dalpha = sample_ratio * label.sum(axis = 0) + pri_alpha - alpha
    dbeta = sample_ratio * label.sum(axis = 0) + pri_beta - beta
    dgamma = sample_ratio * label.T @ X - gamma
    
    return dalpha, dbeta, dgamma

def _calc_objective_function(est_phi, est_label,
                            est_alpha, est_beta, est_gamma,
                            pri_alpha, pri_beta):
    """
    Calculate objective function
    """
    K, M = est_gamma.shape
    est_mean = est_gamma/est_beta[:,np.newaxis]
    
    KL_a = ((est_alpha - pri_alpha) * (psi(est_alpha) - psi(est_alpha.sum()))).sum() \
    - gammaln(est_alpha).sum()+gammaln(est_alpha.sum()) + K*gammaln(pri_alpha) - gammaln(K*pri_alpha)
    KL_b = (M*pri_beta/est_beta + pri_beta * (est_mean**2).sum(axis = 1) - M + M*np.log(est_beta/pri_beta)).sum()/2

    return (est_label * est_phi).sum() - KL_a - KL_b


# -

# # Batch optmization
# + Here, we first consider about natural gradient descent 

# learning settings
iteration = 400
K = 5
pri_alpha = 0.2
pri_beta = 0.001
learning_seed = 20200920
ln_step = 0.1

# +
# initialization
np.random.seed(learning_seed)

est_alpha = np.random.gamma(shape=1, size=K)
est_beta = np.random.gamma(shape=1, size=K)

# est_alpha = np.random.dirichlet([1]*K)
# est_beta = np.random.dirichlet([1]*K)
est_gamma = np.random.normal(scale = 0.5, size = (K, M))

# + jupyter={"outputs_hidden": true}
for ite in range(iteration):
    # update for local parameters
    (est_phi, est_label) = _e_step(train_X, est_alpha, est_beta, est_gamma)
    
    # update for global parameters
    (dalpha, dbeta, dgamma) = _calc_derivative(train_X, est_label, est_alpha, est_beta, est_gamma, 
                                              pri_alpha, pri_beta, n)
    est_alpha += ln_step*dalpha
    est_beta += ln_step*dbeta
    est_gamma += ln_step*dgamma
    est_mean = est_gamma/est_beta[:,np.newaxis]

    # value of the objective function
    F = _calc_objective_function(est_phi, est_label, est_alpha, est_beta, est_gamma, pri_alpha, pri_beta)
#     KL_a = ((est_alpha - pri_alpha) * (psi(est_alpha) - psi(est_alpha.sum()))).sum() \
#     - gammaln(est_alpha).sum()+gammaln(est_alpha.sum()) + K*gammaln(pri_alpha) - gammaln(K*pri_alpha)
#     KL_b = (M*pri_beta/est_beta + pri_beta * (est_mean**2).sum(axis = 1) - M + M*np.log(est_beta/pri_beta)).sum()/2

#     F = (est_label * est_phi).sum() - KL_a - KL_b
    
    print(F, est_alpha)
# -

# # Minibatch Optimiazation

# +
# learning settings
iteration = 40
K = 14
pri_alpha = 0.2
pri_beta = 0.001
learning_seed = 20190920

minibatch_size = 100

alpha = 0.5
beta = 1

# +
# initialization
np.random.seed(learning_seed)

est_alpha = np.random.gamma(shape=1, size=K)
est_beta = np.random.gamma(shape=1, size=K)

# est_alpha = np.random.dirichlet([1]*K)
# est_beta = np.random.dirichlet([1]*K)
est_gamma = np.random.normal(scale = 0.5, size = (K, M))

minibatch_num = n // minibatch_size
# -

n_update = 0
for ite in range(iteration):
    calc_order = np.random.permutation(n)
    est_phi = np.zeros((n,K))
    est_label = np.zeros((n,K))
    for minibatch_ite in range(minibatch_num):
        calc_ind = calc_order[(minibatch_ite*minibatch_size):((minibatch_ite+1)*minibatch_size)] if minibatch_ite < minibatch_num-1 \
        else calc_order[(minibatch_ite*minibatch_size):]
        
        current_X = train_X[calc_ind,:]
        current_step = alpha*(n_update+1)**(-beta)
        
        # update e-step
        (current_phi, current_label) = _e_step(current_X, est_alpha, est_beta, est_gamma)
        est_phi[calc_ind,:] = current_phi
        est_label[calc_ind,:] = current_label

        # update for global parameters
        (dalpha, dbeta, dgamma) = _calc_derivative(current_X, current_label, est_alpha, est_beta, est_gamma, 
                                                  pri_alpha, pri_beta, n)
        est_alpha += current_step*dalpha
        est_beta += current_step*dbeta
        est_gamma += current_step*dgamma
        est_mean = est_gamma/est_beta[:,np.newaxis]        
        
        n_update += 1        
        pass
    F = _calc_objective_function(est_phi, est_label, est_alpha, est_beta, est_gamma, pri_alpha, pri_beta)
    print(F)
    pass

# # Can the GMM by stochastic variational inference capture the change of distribution?
# + By using the above results, we examine the change of the parameter of the GMMM is captured by the SVI.

from sklearn.utils.validation import check_is_fitted


class BayesianGaussianMixtureSVI():
    def __init__(self, K: int,
                 pri_alpha: float = 0.1, pri_beta: float= 0.001,
                 iteration: int = 100, seed: int = -1, minibatch_size: int = 100, tol: float = 1e-5,
                 mu: float = 1, nu: float = 1,
                 is_trace: bool = False, step: int = 10):
        self.K = K
        self.pri_alpha = pri_alpha
        self.pri_beta = pri_beta
        self.iteration = iteration
        self.seed = seed
        self.minibatch_size = minibatch_size
        self.tol = tol
        self.mu = mu
        self.nu = nu
        self.is_trace = is_trace
        self.step = step        
        pass
    
    def _e_step(self, X, alpha, beta, gamma):
        """
        e-step update rule is calculated here.
        In this step, expectation of distribution of local parameters is calculated.
        
        + Input:
            X: (n * M) matrix
            alpha: (K) matrix
            beta: (K) matrix
            gamma: (K * M) matrix
        
        + Output:
            est_phi: (n * K) matrix
                expectation of log likelihood
            est_label: (n * K) matrix
                expectation of local parameters
        
        """

        n, M = X.shape
        K = len(alpha)

        # update for local parameters
        est_phi = np.array([
            psi(alpha) - psi(alpha.sum()) 
            -0.5*((X[i,:].repeat(K).reshape(M,K).T - gamma / beta[:,np.newaxis])**2).sum(axis = 1) 
            - M/2/beta -M/2*np.log(2*np.pi)
        for i in range(n)])
        est_label = softmax(est_phi, axis = 1)

        return est_phi, est_label

    def _calc_derivative(self, X, label, alpha, beta, gamma, 
                         pri_alpha, pri_beta, total_n):
        """
        derivative of the objective function in terms of global parameters
        
        + Input
            X: (n * M) matrix
            label: (n * K) matrix
                expectation of local parameters
            alpha: (K) matrix
            beta: (K) matrix
            gamma: (K * M) matrix
            pri_alpha: scalar, hyperparameter of mixing ratio
            pri_beta: scalar, hyperparameter of center
            total_n: total number of train data
            
        + Output
            Derivative of each global parameters
        """
        n_X = X.shape[0]
        sample_ratio = total_n / n_X

        dalpha = sample_ratio * label.sum(axis = 0) + pri_alpha - alpha
        dbeta = sample_ratio * label.sum(axis = 0) + pri_beta - beta
        dgamma = sample_ratio * label.T @ X - gamma

        return dalpha, dbeta, dgamma

    def _calc_objective_function(self, est_phi, est_label,
                                est_alpha, est_beta, est_gamma,
                                pri_alpha, pri_beta):
        """
        Calculate objective function
        
        
        """
        K, M = est_gamma.shape
        est_mean = est_gamma/est_beta[:,np.newaxis]

        KL_a = ((est_alpha - pri_alpha) * (psi(est_alpha) - psi(est_alpha.sum()))).sum() \
        - gammaln(est_alpha).sum()+gammaln(est_alpha.sum()) + K*gammaln(pri_alpha) - gammaln(K*pri_alpha)
        KL_b = (M*pri_beta/est_beta + pri_beta * (est_mean**2).sum(axis = 1) - M + M*np.log(est_beta/pri_beta)).sum()/2

        return (est_label * est_phi).sum() - KL_a - KL_b
    
    def _initialize(self, K: int, M: int):
        """
        Initialization of global parameters
        
        """
        
        if self.seed > 0:
            np.random.seed(self.seed)

        est_alpha = np.random.gamma(shape=1, size=K)
        est_beta = np.random.gamma(shape=1, size=K)
        est_gamma = np.random.normal(scale = 0.5, size = (K, M))
        
        self.alpha_ = est_alpha
        self.beta_ = est_beta
        self.gamma_ = est_gamma
        pass

    def fit_minibatch(self, X, y=None):
        """
        Learning based on Stochastic Variational Inference
        """
        n, M = X.shape
        K = self.K
        pri_alpha = self.pri_alpha
        pri_beta = self.pri_beta
        iteration = self.iteration
        minibatch_size = self.minibatch_size
        mu = self.mu
        nu = self.nu
        
        # initialization
        if not hasattr(self, "alpha_"):
            self._initialize(K, M)
        est_alpha = self.alpha_
        est_beta = self.beta_
        est_gamma = self.gamma_

        minibatch_num = n // minibatch_size
        n_update = 0
        obj_val = -np.inf
        obj_vals = []
        for ite in range(iteration):
            calc_order = np.random.permutation(n)
            est_phi = np.zeros((n,K))
            est_label = np.zeros((n,K))
            for minibatch_ite in range(minibatch_num):
                calc_ind = calc_order[(minibatch_ite*minibatch_size):((minibatch_ite+1)*minibatch_size)] if minibatch_ite < minibatch_num-1 \
                else calc_order[(minibatch_ite*minibatch_size):]

                current_X = X[calc_ind,:]
                current_step = mu*(n_update+1)**(-nu)

                # update e-step
                (current_phi, current_label) = self._e_step(current_X, est_alpha, est_beta, est_gamma)
                est_phi[calc_ind,:] = current_phi
                est_label[calc_ind,:] = current_label

                # update for global parameters
                (dalpha, dbeta, dgamma) = self._calc_derivative(current_X, current_label, est_alpha, est_beta, est_gamma, 
                                                          pri_alpha, pri_beta, n)
                est_alpha += current_step*dalpha
                est_beta += current_step*dbeta
                est_gamma += current_step*dgamma

                if n_update % self.step == 0:
                    current_obj_val = self._calc_objective_function(est_phi, est_label, est_alpha, est_beta, est_gamma, pri_alpha, pri_beta)
                    obj_vals.append(current_obj_val)
                    if self.is_trace:
                        print(current_obj_val)
                n_update += 1
                pass
            
            current_obj_val = self._calc_objective_function(est_phi, est_label, est_alpha, est_beta, est_gamma, pri_alpha, pri_beta)
            obj_vals.append(current_obj_val)
            if np.abs(obj_val - current_obj_val) < self.tol:
                break
            obj_val = current_obj_val
            
            pass
        est_mean = est_gamma/est_beta[:,np.newaxis]        
        est_ratio = est_alpha / est_alpha.sum()

        self.alpha_ = est_alpha
        self.beta_ = est_beta
        self.gamma_ = est_gamma
        self.ratio_ = est_ratio
        self.mean_ = est_mean
        if self.is_trace:
            self.obj_vals_ = obj_vals
        return self


# data settings
n = 10000
M = 5 # data dimenension
data_seed = 20200920
np.random.seed(data_seed)

# data generation
K0 = 3
true_ratio = np.random.dirichlet([1]*K0)
true_mean = np.random.normal(size = (K0, M), scale = 2)

train_X1, _, data_label = GaussianMixtureModel.rvs(true_ratio, true_mean, precision = np.ones((K0, M)), size = n)

# data generation
K0 = 5
true_ratio = np.random.dirichlet([1]*K0)
true_mean = np.random.normal(size = (K0, M), scale = 2)

train_X2, _, data_label = GaussianMixtureModel.rvs(true_ratio, true_mean, precision = np.ones((K0, M)), size = n)

ln_params = {
    "K": 8,
    "pri_alpha": 0.1,
    "pri_beta": 0.001,
    "seed": 20200920,
    "iteration": 10,
    "minibatch_size": 1000,
    "is_trace": True,
    "step": 100
}

ln_obj = BayesianGaussianMixtureSVI(**ln_params)

ln_obj.fit_minibatch(train_X1)

ln_obj.ratio_

ln_obj.fit_minibatch(train_X2)

ln_obj.ratio_

# +
# learning settings
iteration = 40
K = 8
pri_alpha = 0.2
pri_beta = 0.001
learning_seed = 20190920

minibatch_size = 100

alpha = 0.5
beta = 1

# +
# initialization
np.random.seed(learning_seed)

est_alpha = np.random.gamma(shape=1, size=K)
est_beta = np.random.gamma(shape=1, size=K)

# est_alpha = np.random.dirichlet([1]*K)
# est_beta = np.random.dirichlet([1]*K)
est_gamma = np.random.normal(scale = 0.5, size = (K, M))

minibatch_num = n // minibatch_size
# -

n_update = 0
for ite in range(iteration):
    calc_order = np.random.permutation(n)
    est_phi = np.zeros((n,K))
    est_label = np.zeros((n,K))
    for minibatch_ite in range(minibatch_num):
        calc_ind = calc_order[(minibatch_ite*minibatch_size):((minibatch_ite+1)*minibatch_size)] if minibatch_ite < minibatch_num-1 \
        else calc_order[(minibatch_ite*minibatch_size):]
        
        current_X = train_X[calc_ind,:]
        current_step = alpha*(n_update+1)**(-beta)
        
        # update e-step
        (current_phi, current_label) = _e_step(current_X, est_alpha, est_beta, est_gamma)
        est_phi[calc_ind,:] = current_phi
        est_label[calc_ind,:] = current_label

        # update for global parameters
        (dalpha, dbeta, dgamma) = _calc_derivative(current_X, current_label, est_alpha, est_beta, est_gamma, 
                                                  pri_alpha, pri_beta, n)
        est_alpha += current_step*dalpha
        est_beta += current_step*dbeta
        est_gamma += current_step*dgamma
        est_mean = est_gamma/est_beta[:,np.newaxis]        
        
        n_update += 1        
        pass
    F = _calc_objective_function(est_phi, est_label, est_alpha, est_beta, est_gamma, pri_alpha, pri_beta)
    print(F)
    pass




