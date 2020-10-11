# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# # Linear regression with laplace prior
#  + In general, laplace prior gives sparse result for regression
#      + However, it is difficult to deal with it well due to non-differential point at the origin.
#          + $\log p(w) \equiv -1/\beta \sum_j |w_j| $, $|w_j|$ is non-differential at the origin.
#  + By the way, non-differential point is eliminated by integrating $|w_j|$:
#      + $E[|w_j|]$ does not have non-diffenrential point when the distribution is normal distribution.
#      + It is achieved when we consider about the objective function of variational Bayes.
#          + $\mathcal{F} := E[\log \frac{q(w)}{p(Y|X,w}p(w)]$
#          + Here, $\mathcal{F}$ has a parameter that decides the form of $q(w) = N(w|m, \Sigma)$, $(m, \Sigma)$ is the parameter and optimized by it.
#  + In this notebook, the approximated posterior distribution by Variational Bayes is studied.
#      + The objective function is optimized by a gradient descent method.
#          + Specifically, the Natural gradient descent is efficient method when we consider about a constrained parameter like positive definite matrix, positive real value, simplex, and so on.
#          + Thus, we used the natural gradient descent.

# # Formulation
# + Learning Model:
#     + $p(y|x,w) = N(y|x \cdot w, 1), y \in mathbb{R}, x,w \in \mathbb{R}^M$
#     + $p(w) \equiv \exp(-\frac{1}{\beta} \sum_j |w_j|)$, $\beta$ is hyperparameter.
# + Approximated Variational Posterior distribution:
#     + $q(w) = N(w|m, \Sigma)$
#         + $m \in \mathbb{R}^M, \Sigma \in \mathbb{R}^{M \times M}$ is the parameters to be optimized.

# # In this notebook
# + We compare the following average generalization error:
# $$
#     G(n) = \frac{1}{L} \sum_{j=1}^L \| y - X \hat{w}(x^l, y^l) \|^2,
# $$
# where $\hat{w}$ is estimated parameter by $(x^l, y^l)$.  
# We evaluate the error among Lasso, Ridge, and VB laplace(this calculation).

# # Preliminary
# ## Import library

import numpy as np
from scipy.stats import norm
from scipy.stats import invwishart

from sklearn.linear_model import LassoCV, Lasso, LassoLarsCV
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.base import BaseEstimator, RegressorMixin

# ## Data setting

# +
## data setting
n = 100 # train size
M = 400 # # of features
n_zero_ind = M // 4 * 3 # # of zero elements in the parameter
prob_seed = 20201110 # random seed

N = 10000 # test size

datasets = 100
# -

# ## Problem setting

np.random.seed(prob_seed)
true_w = np.random.normal(scale = 3, size = M)
zero_ind = np.random.choice(M, size = n_zero_ind)
true_w[zero_ind] = 0

# ## Learning settings

ln_vb_params = {
    "pri_beta": 10,
    "pri_opt_flag": True,
    "iteration": 10000,
    "step": 0.2,
    "is_trace": False,
    "trace_step": 100
}
ln_lasso_params = {
    "fit_intercept": False,
    "cv": 5,
    "max_iter": 10000
}
ln_ridge_params = {
    "fit_intercept": False,
    "cv": 5
}


# ## Classes

class VBLaplace(BaseEstimator, RegressorMixin):
    def __init__(
        self, pri_beta: float = 20, pri_opt_flag: bool = True,
        seed: int = -1, iteration: int = 1000, tol: float = 1e-8, step: float = 0.1,
        is_trace: bool = False, trace_step: int = 20
    ):
        self.pri_beta = pri_beta
        self.pri_opt_flag = pri_opt_flag
        self.seed = seed
        self.iteration = iteration
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        self.trace_step = trace_step
        pass
    
    def _initialization(self, M: int):
        seed = self.seed
        
        if seed > 0:
            np.random.seed(seed)
        
        mean = np.random.normal(size = M)
        sigma = invwishart.rvs(df = M+2, scale = np.eye(M), size = 1)
        pri_beta = np.random.gamma(shape = 3, size = 1) if self.pri_opt_flag else self.pri_beta
        
        self.mean_ = mean
        self.sigma_ = sigma
        self.pri_beta_ = pri_beta
        pass
    
    def _obj_func(self, X:np.ndarray, y:np.ndarray, pri_beta:float, mean:np.ndarray, sigma:np.ndarray) -> float:
        """
        Calculate objective function.

        + Input:
            1. X: input matrix (n, M) matrix
            2. y: output vector (n, ) matrix
            3. mean: mean parameter of vb posterior
            4. sigma: covariance matrix of vb posterior

        + Output:
            value of the objective function.

        """

        n, M = X.shape

        sq_sigma_diag = np.sqrt(np.diag(sigma))
        log_2pi = np.log(2*np.pi)

        F = 0
        # const values
        F += -M/2*log_2pi -M/2 + M*log_2pi + n*M/2*log_2pi + M*np.log(2*pri_beta)

        F += ((y - X@mean)**2).sum()/2 - np.linalg.slogdet(sigma)[1]/2 + np.trace(X.T @ X @ sigma)/2

        # term obtained from laplace prior
        F += ((mean + 2*sq_sigma_diag*norm.pdf(-mean/sq_sigma_diag)-2*mean*norm.cdf(-mean/sq_sigma_diag))/pri_beta).sum()

        return F
    
    def fit(self, train_X:np.ndarray, train_Y:np.ndarray):
        pri_beta = self.pri_beta
        iteration = self.iteration
        step = self.step
        tol = self.tol
        
        is_trace = self.is_trace
        trace_step = self.trace_step
        
        M = train_X.shape[1]
        
        if not hasattr(self, "mean_"):
            self._initialization(M)
        
        est_mean = self.mean_
        est_sigma = self.sigma_
        est_pri_beta = self.pri_beta_
        
        # transformation to natural parameter
        theta1 = np.linalg.solve(est_sigma, est_mean)
        theta2 = -np.linalg.inv(est_sigma)/2        
        
        F = []
        for ite in range(iteration):
            sq_sigma_diag = np.sqrt(np.diag(est_sigma))

            # update mean and sigma by natural gradient
            dFdnu1 = theta1 - train_Y @ train_X
            dFdnu1 += (1 - 2*est_mean/sq_sigma_diag*norm.pdf(-est_mean/sq_sigma_diag) - 2*norm.cdf(-est_mean/sq_sigma_diag)) / est_pri_beta
            dFdnu2 = theta2 + train_X.T @ train_X/2
            dFdnu2[np.diag_indices(M)] += 1/sq_sigma_diag*norm.pdf(-est_mean/sq_sigma_diag)/est_pri_beta

            theta1 += -step * dFdnu1
            theta2 += -step * dFdnu2
            est_sigma = -np.linalg.inv(theta2)/2
            est_mean = est_sigma @ theta1
            
            # update pri_beta by extreme value
            est_pri_beta = ((est_mean + 2*sq_sigma_diag*norm.pdf(-est_mean/sq_sigma_diag)-2*est_mean*norm.cdf(-est_mean/sq_sigma_diag))).mean() if self.pri_opt_flag else pri_beta
            current_F = self._obj_func(train_X, train_Y, est_pri_beta, est_mean, est_sigma)
            if is_trace and ite % trace_step == 0:
                print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())            
            
            if ite > 0 and np.abs(current_F - F[ite-1]) < tol:
                if is_trace:
                    print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())                            
                break
            else:
                F.append(current_F)
            pass
        
        
        self.F_ = F
        self.mean_ = est_mean
        self.sigma_ = est_sigma
        self.pri_beta_ = est_pri_beta
        
        return self
        pass
    
    def predict(self, test_X: np.ndarray):
        if not hasattr(self, "mean_"):
            raise ValueError("fit has not finished yet, should fit before predict.")
        return test_X @ self.mean_
        pass
        
    pass


class VBNormal(BaseEstimator, RegressorMixin):
    def __init__(
        self, pri_beta: float = 20, pri_opt_flag: bool = True,
        seed: int = -1, iteration: int = 1000, tol: float = 1e-8, step: float = 0.1,
        is_trace: bool = False, trace_step: int = 20
    ):
        self.pri_beta = pri_beta
        self.pri_opt_flag = pri_opt_flag
        self.seed = seed
        self.iteration = iteration
        self.tol = tol
        self.step = step
        self.is_trace = is_trace
        self.trace_step = trace_step
        pass
    
    def _initialization(self, M: int):
        seed = self.seed
        
        if seed > 0:
            np.random.seed(seed)
        
        mean = np.random.normal(size = M)
        sigma = invwishart.rvs(df = M+2, scale = np.eye(M), size = 1)
        pri_beta = np.random.gamma(shape = 3, size = 1) if self.pri_opt_flag else self.pri_beta
        
        self.mean_ = mean
        self.sigma_ = sigma
        self.pri_beta_ = pri_beta
        pass
    
    def _obj_func(self, X:np.ndarray, y:np.ndarray, pri_beta:float, mean:np.ndarray, sigma:np.ndarray) -> float:
        """
        Calculate objective function.

        + Input:
            1. X: input matrix (n, M) matrix
            2. y: output vector (n, ) matrix
            3. mean: mean parameter of vb posterior
            4. sigma: covariance matrix of vb posterior

        + Output:
            value of the objective function.

        """

        n, M = X.shape

        log_2pi = np.log(2*np.pi)

        F = 0
        # const values
        F += -M/2*log_2pi -M/2 + M*log_2pi + n*M/2*log_2pi + M*np.log(2*pri_beta)

        F += ((y - X@mean)**2).sum()/2 - np.linalg.slogdet(sigma)[1]/2 + np.trace(X.T @ X @ sigma)/2

        # term obtained from Normal prior
        F += pri_beta/2*(mean@mean + np.trace(sigma)) - M/2*np.log(pri_beta) + M/2*log_2pi
        
        return F
    
    def fit(self, train_X:np.ndarray, train_Y:np.ndarray):
        pri_beta = self.pri_beta
        iteration = self.iteration
        step = self.step
        tol = self.tol
        
        is_trace = self.is_trace
        trace_step = self.trace_step
        
        M = train_X.shape[1]
        
        if not hasattr(self, "mean_"):
            self._initialization(M)
        
        est_mean = self.mean_
        est_sigma = self.sigma_
        est_pri_beta = self.pri_beta_
                
        F = []
        X_cov = train_X.T@train_X
        
        for ite in range(iteration):
            sigma_inv = X_cov + est_pri_beta*np.eye(M)
            est_mean = np.linalg.solve(sigma_inv, train_Y@train_X)
            est_sigma = np.linalg.inv(sigma_inv)
            
            # update pri_beta by extreme value
            est_pri_beta = M/(est_mean@est_mean + np.trace(est_sigma)) if self.pri_opt_flag else pri_beta
            current_F = self._obj_func(train_X, train_Y, est_pri_beta, est_mean, est_sigma)
            if is_trace and ite % trace_step == 0:
                print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())            
            
            if ite > 0 and np.abs(current_F - F[ite-1]) < tol:
                if is_trace:
                    print(current_F, (dFdnu1**2).sum(), (dFdnu2**2).sum())                            
                break
            else:
                F.append(current_F)
            pass
        
        
        self.F_ = F
        self.mean_ = est_mean
        self.sigma_ = est_sigma
        self.pri_beta_ = est_pri_beta
        
        return self
        pass
    
    def predict(self, test_X: np.ndarray):
        if not hasattr(self, "mean_"):
            raise ValueError("fit has not finished yet, should fit before predict.")
        return test_X @ self.mean_
        pass
        
    pass

# # Experiment part
# + By some datasets are used for train and evaluate

# +
sq_error_vb_laplace = np.zeros(datasets)
sq_error_vb_normal = np.zeros(datasets)
sq_error_lasso = np.zeros(datasets)
sq_error_ridge = np.zeros(datasets)

for dataset_ind in range(datasets):
    vb_laplace_obj = VBLaplace(**ln_vb_params)
    vb_normal_obj = VBNormal(**ln_vb_params)
    lasso_obj = LassoCV(**ln_lasso_params)
    ridge_obj = RidgeCV(**ln_ridge_params)    
    # data generation
    train_X = np.random.normal(size = (n, M))
    train_Y = train_X @ true_w + np.random.normal(size = n)

    lasso_obj.fit(train_X, train_Y)
    ridge_obj.fit(train_X, train_Y)
    vb_laplace_obj.fit(train_X, train_Y)
    vb_normal_obj.fit(train_X, train_Y)

    test_X = np.random.normal(size = (N, M))
    test_Y = test_X @ true_w + np.random.normal(size = N)
    
    sq_error_lasso[dataset_ind] = ((test_X @ lasso_obj.coef_- test_Y)**2).mean()
    sq_error_ridge[dataset_ind] = ((test_X @ ridge_obj.coef_- test_Y)**2).mean()
    sq_error_vb_laplace[dataset_ind] = ((test_X @ vb_laplace_obj.mean_- test_Y)**2).mean()
    sq_error_vb_normal[dataset_ind] = ((test_X @ vb_normal_obj.mean_- test_Y)**2).mean()
    print(
        sq_error_lasso[dataset_ind]
        , sq_error_ridge[dataset_ind]
        , sq_error_vb_laplace[dataset_ind]
        , sq_error_vb_normal[dataset_ind]
    )
# -

print(
    sq_error_lasso.mean()
    , sq_error_ridge.mean()
    , sq_error_vb_laplace.mean()
    , sq_error_vb_normal.mean()
)

sq_error_lasso.std()

# + {"jupyter": {"outputs_hidden": true}}
true_w
# -








