# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
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

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import invwishart

from scipy.optimize import minimize
from sklearn.linear_model import LassoCV, Lasso, LassoLarsCV
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import ARDRegression
from sklearn.base import BaseEstimator, RegressorMixin

# ## Data setting

# +
## data setting
n = 200 # train size
M = 200 # # of features
zero_ratio = 0.5
n_zero_ind = int(M*zero_ratio) # # of zero elements in the parameter
prob_seed = 20210112 # random seed

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
ln_ard_params = {
    "fit_intercept": False
}
ln_vb_post_laplace_params = {
    "pri_beta": 1,
    "iteration": 1000,
    "is_trace": False
}


# # Classes

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
        
        cov_X = train_X.T @ train_X
        cov_YX = train_Y @ train_X
        for ite in range(iteration):
            sq_sigma_diag = np.sqrt(np.diag(est_sigma))

            # update mean and sigma by natural gradient
            dFdnu1 = theta1 - cov_YX
            dFdnu1 += (1 - 2*est_mean/sq_sigma_diag*norm.pdf(-est_mean/sq_sigma_diag) - 2*norm.cdf(-est_mean/sq_sigma_diag)) / est_pri_beta
            dFdnu2 = theta2 + cov_X/2
            dFdnu2[np.diag_indices(M)] += 1/sq_sigma_diag*norm.pdf(-est_mean/sq_sigma_diag)/est_pri_beta

            theta1 += -step * dFdnu1
            theta2 += -step * dFdnu2
            est_sigma = -np.linalg.inv(theta2)/2
            est_mean = est_sigma @ theta1
            
            # update pri_beta by extreme value
            sq_sigma_diag = np.sqrt(np.diag(est_sigma))
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
        XY_cov = train_Y@train_X
        X_cov = train_X.T@train_X
        
        for ite in range(iteration):
            sigma_inv = X_cov + est_pri_beta*np.eye(M)
            est_mean = np.linalg.solve(sigma_inv, XY_cov)
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


class VBApproxLaplace(BaseEstimator, RegressorMixin):
    """
    Laplace prior is approximated by normal distribution, and approximated posterior distribution is obtained by the approximated laplace prior.
    """
    
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
    
    def _obj_func(self, y:np.ndarray, pri_beta:float, mean:np.ndarray, inv_sigma:np.ndarray, h_xi: np.ndarray, v_xi: np.ndarray) -> float:
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

        F = 0
        F += pri_beta/2*np.sqrt(h_xi).sum() + v_xi@h_xi - M*np.log(pri_beta/2)
        F += n/2*np.log(2*np.pi) + train_Y@train_Y/2 - mean @ (inv_sigma @ mean)/2 + np.linalg.slogdet(inv_sigma)[0]/2
        return F
    
    def fit(self, train_X:np.ndarray, train_Y:np.ndarray):
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
        XY_cov = train_X.T @ train_Y
        
        for ite in range(iteration):
            # update form of approximated laplace prior
            est_h_xi = est_mean**2 + np.diag(est_sigma)
            est_v_xi = -est_pri_beta/2/np.sqrt(est_h_xi)            
            
            # update posterior distribution
            inv_sigma = X_cov -2*np.diag(est_v_xi)
            est_mean = np.linalg.solve(inv_sigma, XY_cov)
            est_sigma = np.linalg.inv(inv_sigma)
            
            # update pri_beta by extreme value
            est_pri_beta = M/((est_mean**2 + np.diag(est_sigma))/(2*np.sqrt(est_h_xi))).sum() if self.pri_opt_flag else pri_beta
            
            current_F = self._obj_func(train_Y, est_pri_beta, est_mean, inv_sigma, est_h_xi, est_v_xi)
            if is_trace and ite % trace_step == 0:
                print(current_F)            
            
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


class LaplacePosteriorVB(BaseEstimator, RegressorMixin):

    def __init__(self, pri_beta: float = 0.1, seed: int = -1, is_pri_optimize: bool = False, iteration: int = 1000, is_trace: bool = True):
        """
        LaplacePosteriorVB is a class to calculate an approximated posterior distribution in the diagonal Laplace distribution q(w|lambda):
        
        q(w|lambda=(mu,sigma)) = prod_j=1^M sqrt{2}/sigma_j exp(-sqrt{2}/sigma_j |w_j-mu_j|),
        where mu in mathbb{R}^M, sigma in mathbb{R}^M.
        
        The method searches an optimized posterior in terms of minimizing KL(q(w)||p(w|X^n,Y^n)),
        where p(w|X^n,Y^n) propto p(Y|w,X) p(w), and p(Y|w,X)=N(Y|Xw, I_n), p(w)=q(w|0_M, pri_beta),
        i.e. we consider here ordinal linear regression problem.        
        
        """
        self.pri_beta = pri_beta
        self.seed = seed
        self.iteration = iteration
        self.is_trace = is_trace
        pass
    
    def _initialize(self) -> (np.ndarray, np.ndarray, float):
        """
        Initialize parameters for an approximated posterior distribution.
        """
        if self.seed > 0:
            np.random.seed(self.seed)
        
        est_mu = np.random.normal(size = M)
        est_ln_sigma = np.random.normal(size = M)
        est_pri_beta = self.pri_beta
        return est_mu, est_ln_sigma, est_pri_beta
        pass
    
    def _calc_energy(self, post_mu: np.ndarray, post_ln_sigma: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> float:
        """
        Objective function over parameters for the posterior.
        """
        
        post_sigma = np.exp(post_ln_sigma)
        n, M = X.shape
        energy = 0
        energy += ((y-X@post_mu)**2).sum()/2 + (X**2).sum(axis=0)@post_sigma**2/2 + n/2*np.log(2*np.pi) - M - M*np.log(pri_beta)
        energy += (-np.log(post_sigma) + np.sqrt(2)/pri_beta*np.abs(post_mu) + post_sigma/pri_beta*np.exp(-np.sqrt(2)/post_sigma*np.abs(post_mu))).sum()    
        return energy    
    
    def _calc_energy_wrapper(self, est_params: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> np.ndarray:
        post_mu = est_params[:M]
        post_ln_sigma = est_params[M:]
        return self._calc_energy(post_mu, post_ln_sigma, X, y, pri_beta)
        pass    
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        
        (est_mu, est_ln_sigma, est_pri_beta) = self._initialize()
        res = minimize(
            fun=self._calc_energy_wrapper, x0=np.hstack([est_mu, est_ln_sigma]), 
            args=(train_X, train_Y, est_pri_beta), method = "L-BFGS-B", options={"disp":self.is_trace, "maxiter": self.iteration}
        )
        
        est_mu = res.x[:M]
        est_ln_sigma = res.x[M:]
        est_sigma = np.exp(est_ln_sigma)
        
        self.mu_ = est_mu
        self.sigma_ = est_sigma
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "mu_")
        return X@self.mu_
        pass

    def get_params(self, deep=True) -> dict:
        return {'pri_beta': self.pri_beta}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
    
    pass


# # Experiment part
# + By some datasets are used for train and evaluate

score_func = lambda X, y, coef: 1 - ((y - X@coef)**2).sum() / ((y - y.mean())**2).sum()
score_vb_laplace_exact = np.zeros(datasets)
score_vb_laplace_approx = np.zeros(datasets)
score_vb_normal = np.zeros(datasets)
score_vb_post_laplace = np.zeros(datasets)
score_ard = np.zeros(datasets)
score_lasso = np.zeros(datasets)
score_ridge = np.zeros(datasets)

sq_error = lambda X, y, coef: ((y - X@coef)**2).mean()
sq_error_vb_laplace_exact = np.zeros(datasets)
sq_error_vb_laplace_approx = np.zeros(datasets)
sq_error_vb_normal = np.zeros(datasets)
sq_error_vb_post_laplace = np.zeros(datasets)
sq_error_ard = np.zeros(datasets)
sq_error_lasso = np.zeros(datasets)
sq_error_ridge = np.zeros(datasets)

for dataset_ind in range(datasets):
    vb_laplace_exact_obj = VBLaplace(**ln_vb_params)
    vb_laplace_approx_obj = VBApproxLaplace(**ln_vb_params)
    vb_normal_obj = VBNormal(**ln_vb_params)
    vb_post_laplace_obj = LaplacePosteriorVB(**ln_vb_post_laplace_params)
    lasso_obj = LassoCV(**ln_lasso_params)
    ridge_obj = RidgeCV(**ln_ridge_params)
    ard_obj = ARDRegression(**ln_ard_params)
    
    # data generation
    train_X = np.random.normal(size = (n, M))
    train_Y = train_X @ true_w + np.random.normal(size = n)

    lasso_obj.fit(train_X, train_Y)
    ridge_obj.fit(train_X, train_Y)
    ard_obj.fit(train_X, train_Y)
    vb_laplace_exact_obj.fit(train_X, train_Y)
    vb_normal_obj.fit(train_X, train_Y)
    vb_laplace_approx_obj.fit(train_X, train_Y)
    vb_post_laplace_obj.fit(train_X, train_Y)

    test_X = np.random.normal(size = (N, M))
    test_Y = test_X @ true_w + np.random.normal(size = N)
    
    ### evaluation by square error
    sq_error_lasso[dataset_ind] = sq_error(test_X, test_Y, lasso_obj.coef_)
    sq_error_ridge[dataset_ind] = sq_error(test_X, test_Y, ridge_obj.coef_)
    sq_error_ard[dataset_ind] = sq_error(test_X, test_Y, ard_obj.coef_)
    sq_error_vb_laplace_exact[dataset_ind] = sq_error(test_X, test_Y, vb_laplace_exact_obj.mean_)
    sq_error_vb_normal[dataset_ind] = sq_error(test_X, test_Y, vb_normal_obj.mean_)
    sq_error_vb_laplace_approx[dataset_ind] = sq_error(test_X, test_Y, vb_laplace_approx_obj.mean_)
    sq_error_vb_post_laplace[dataset_ind] = sq_error(test_X, test_Y, vb_post_laplace_obj.mu_)

    print(
        "sq_error:"
        , sq_error_lasso[dataset_ind]
        , sq_error_ridge[dataset_ind]
        , sq_error_ard[dataset_ind]
        , sq_error_vb_laplace_exact[dataset_ind]
        , sq_error_vb_normal[dataset_ind]
        , sq_error_vb_laplace_approx[dataset_ind]
        , sq_error_vb_post_laplace[dataset_ind]
    )    
    
    ### evaluation by R^2 score
    score_lasso[dataset_ind] = score_func(test_X, test_Y, lasso_obj.coef_)
    score_ridge[dataset_ind] = score_func(test_X, test_Y, ridge_obj.coef_)
    score_ard[dataset_ind] = score_func(test_X, test_Y, ard_obj.coef_)
    score_vb_laplace_exact[dataset_ind] = score_func(test_X, test_Y, vb_laplace_exact_obj.mean_)
    score_vb_normal[dataset_ind] = score_func(test_X, test_Y, vb_normal_obj.mean_)
    score_vb_laplace_approx[dataset_ind] = score_func(test_X, test_Y, vb_laplace_approx_obj.mean_)
    score_vb_post_laplace[dataset_ind] = score_func(test_X, test_Y, vb_post_laplace_obj.mu_)
    
    print(
        "R^2 score:"
        , score_lasso[dataset_ind]
        , score_ridge[dataset_ind]
        , score_ard[dataset_ind]
        , score_vb_laplace_exact[dataset_ind]
        , score_vb_normal[dataset_ind]
        , score_vb_laplace_approx[dataset_ind]
        , score_vb_post_laplace[dataset_ind]
    )


print(
    sq_error_lasso.mean()
    , sq_error_ridge.mean()
    , sq_error_ard.mean()
    , sq_error_vb_laplace_exact.mean()
    , sq_error_vb_normal.mean()
    , sq_error_vb_laplace_approx.mean()
)

print(
    score_lasso.mean()
    , score_ridge.mean()
    , score_ard.mean()
    , score_vb_laplace_exact.mean()
    , score_vb_normal.mean()
    , score_vb_laplace_approx.mean()
)

np.abs(vb_laplace_exact_obj.mean_ - np.sqrt(np.diag(vb_laplace_exact_obj.sigma_)))

upper = vb_laplace_exact_obj.mean_ + 0.8 * np.sqrt(np.diag(vb_laplace_exact_obj.sigma_))

lower = vb_laplace_exact_obj.mean_ - 0.8 * np.sqrt(np.diag(vb_laplace_exact_obj.sigma_))

(((lower < 0) & (0 < upper)))[:100]

(np.abs(vb_laplace_exact_obj.mean_) < 1).sum()

(lasso_obj.coef_ < 0.001).sum()

(true_w < 0.001).sum()

import matplotlib.pyplot as plt

# + {"jupyter": {"outputs_hidden": true}}
true_w
# -




# # Conclusion
# + We experimented the performance of the rigorously derived variational linear regression algorithm for the Laplace prior by comparing:
#     1. Ordinal optimized Lasso by cross-validation
#     2. Ordinal optimized Ridge by cross-validation
#     3. Variational Bayes linear regression for the normal prior
#     4. Bayesian ARD
#     5. Variational Bayes linear regression for the approximated Laplace prior.
# + Results are as follows:
#     1. n > M with non-zero elements: ridge, vb for the normal prior gives the best performance, although vb for the Laplace prior gives better performance.
#     2. n > M with zero-elements: lasso, vb for the approximated Laplace gives the best performance. although vb for the Laplace prior also gives better performance.
#     3. M > n with zero-elements: results is similar with 1.
#     4. M > n with zero-elements: results is similar with 2.
#     5. M >> n, especially # of non-zero elements is larger than # of samples, vb for the Laplace prior gives the best performance.
# + Summary of results:
#     + Derived algorithm can estimate every case, and # of features are extremely larger. 



