# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: study
#     language: python
#     name: study
# ---

# # VB Lasso by Laplace posterior
# + We consider the following approximated posterior distribution q(w):
# $$
# p(y|x,w) = N(y|x^T w,1), p(w) = Laplace(w, 0, \beta)  \\
# q(w) \propto \prod_{j=1}^M \exp(-\frac{1}{\sigma_j} |w_j - \mu_j|),
# $$
# where $\sigma_j \in \mathbb{R}_+, \mu_j, w_j \in \mathbb{R}$

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV, Lasso
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

# +
# mu = -1
# sigma = 0.1

# val = np.random.laplace(loc = mu, scale = sigma/np.sqrt(2), size = 10000)

# (1/sigma**2 - 2*np.sqrt(2)/sigma**3*np.abs(val - mu) + 2/sigma**4*(val - mu)**2).mean()

# 1/sigma**2

# (3-np.sqrt(2))/sigma**2
# -

# ## Problem settings

n = 200
data_seed = 20210112
M = 200
zero_ratio = 0.5
n_zero_ind = int(M*zero_ratio) # # of zero elements in the parameter

# ## Data Generation

np.random.seed(data_seed)

true_w = np.random.normal(scale = 3, size = M)
zero_ind = np.random.choice(M, size = n_zero_ind)
true_w[zero_ind] = 0

# ## Learning settings

iteration = 500
ln_seed = 20210105
np.random.seed(ln_seed)
is_trace = False
pri_beta = 1.5
rho = 0.1


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
        est_pri_beta = pri_beta
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


# +
# ### initialization
# est_mu = np.random.normal(size = M)
# est_ln_sigma = np.random.normal(size = M)
# est_sigma = np.exp(est_ln_sigma)
# est_pri_beta = pri_beta

# def calc_energy(post_mu: np.ndarray, post_ln_sigma: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> float:
#     post_sigma = np.exp(post_ln_sigma)
#     n, M = X.shape
#     energy = 0
#     energy += ((y-X@post_mu)**2).sum()/2 + (X**2).sum(axis=0)@post_sigma**2/2 + n/2*np.log(2*np.pi) - M - M*np.log(pri_beta)
#     energy += (-np.log(post_sigma) + np.sqrt(2)/pri_beta*np.abs(post_mu) + post_sigma/pri_beta*np.exp(-np.sqrt(2)/post_sigma*np.abs(post_mu))).sum() 
# #     energy += (-np.log(post_sigma) + np.sqrt(2)/pri_beta*np.abs(post_mu) + post_sigma/pri_beta*np.exp(-post_sigma/np.sqrt(2)*np.abs(post_mu))).sum()    
#     return energy

# def calc_energy_dash(post_mu: np.ndarray, post_ln_sigma: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> float:
#     post_sigma = np.exp(post_ln_sigma)
#     n, M = X.shape
#     energy = 0
#     energy += ((y-X@post_mu)**2).sum()/2 + (X**2).sum(axis=0)@post_sigma**2/2 + n/2*np.log(2*np.pi) - n*M - n*M*np.log(pri_beta)
#     energy += n*(-np.log(post_sigma) + np.sqrt(2)/pri_beta*np.abs(post_mu) + post_sigma/pri_beta*np.exp(-post_sigma/np.sqrt(2)*np.abs(post_mu))).sum()
#     return energy

# def calc_energy_wrapper(est_params: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> np.ndarray:
#     post_mu = est_params[:M]
#     post_ln_sigma = est_params[M:]
#     return calc_energy(post_mu, post_ln_sigma, X, y, pri_beta)
#     pass

# def df_param(X :np.ndarray, y :np.ndarray, mu: np.ndarray, sigma: np.ndarray, pri_beta: float):
#     pdf_mu = np.exp(-sigma/np.sqrt(2)*np.abs(mu))*sigma/pri_beta
#     dFdm = -X.T @ (y - X @ mu) + np.sqrt(2)/pri_beta*np.sign(mu) - pdf_mu*sigma/np.sqrt(2)*np.sign(mu) 
#     dFds = (X**2).sum(axis = 0)*sigma - 1/sigma + pdf_mu*(1-sigma/np.sqrt(2)*np.abs(mu))
    
#     return dFdm, dFds

# def df_param_dash(X :np.ndarray, y :np.ndarray, mu: np.ndarray, sigma: np.ndarray, pri_beta: float):
#     n = len(y)
#     pdf_mu = np.exp(-sigma/np.sqrt(2)*np.abs(mu))*sigma/pri_beta
#     dFdm = -X.T @ (y - X @ mu) + n*np.sqrt(2)/pri_beta*np.sign(mu) - n*pdf_mu*sigma/np.sqrt(2)*np.sign(mu) 
#     dFds = (X**2).sum(axis = 0)*sigma - n/sigma + n*pdf_mu*(1-sigma/np.sqrt(2)*np.abs(mu))
    
#     return dFdm, dFds

# +
train_X = np.random.normal(size = (n, M))
train_Y = train_X @ true_w + np.random.normal(size = n)

test_X = np.random.normal(size = (n, M))
test_Y = test_X @ true_w + np.random.normal(size = n)

# +
# res = minimize(
#     fun=calc_energy_wrapper, x0=np.hstack([est_mu, est_ln_sigma]), 
#     args=(train_X, train_Y, est_pri_beta), method = "L-BFGS-B", options={"disp":True, "maxiter": 1000}
# )
# est_mu = res.x[:M]
# est_ln_sigma = res.x[M:]
# est_sigma = np.exp(est_ln_sigma)

# for ite in range(iteration):
#     res = minimize(
#         fun=calc_energy_wrapper, x0=np.hstack([est_mu, est_ln_sigma]), 
#         args=(train_X, train_Y, est_pri_beta), method = "L-BFGS-B", options={"disp":True, "maxiter": 1}
#     )

#     est_mu = res.x[:M]
#     est_ln_sigma = res.x[M:]
#     est_sigma = np.exp(est_ln_sigma)
#     est_pri_beta = (np.sqrt(2)*np.abs(est_mu) + est_sigma*np.exp(-np.sqrt(2)/est_sigma*np.abs(est_mu))).mean()

#     dFdm, dFds = df_param(train_X, train_Y, est_mu, est_sigma, est_pri_beta)
    
# #     print(res.fun, (dFdm**2).mean(), (dFds**2).mean())

# post_mu = res.x[:M]
# post_sigma = np.exp(res.x[M:])
# -

alphas = [100, 10, 5, 1, 0.5, 0.1, 0.01, 0.05, 0.001]

laplace_vb_obj = GridSearchCV(
    LaplacePosteriorVB(**{
        "pri_beta": pri_beta,
        "iteration": iteration,
        "is_trace": is_trace
    })
    , param_grid = {
        "is_trace": [False],
        "pri_beta": alphas
    }
    , cv=3
)

laplace_vb_obj.fit(train_X, train_Y)

lasso_obj = LassoCV(alphas=alphas, fit_intercept=False, cv=3)
lasso_obj.fit(train_X, train_Y)



lasso_obja

print(np.sqrt(((test_Y - test_X@lasso_obj.coef_)**2).mean()))
print(np.sqrt(((test_Y - test_X@laplace_vb_obj.best_estimator_.mu_)**2).mean()))
print(np.sqrt(((test_Y - test_X@true_w)**2).mean()))


