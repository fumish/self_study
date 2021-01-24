# ---
# jupyter:
#   jupytext:
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

# +
# mu = -1
# sigma = 0.1

# val = np.random.laplace(loc = mu, scale = sigma/np.sqrt(2), size = 10000)

# (1/sigma**2 - 2*np.sqrt(2)/sigma**3*np.abs(val - mu) + 2/sigma**4*(val - mu)**2).mean()

# 1/sigma**2

# (3-np.sqrt(2))/sigma**2
# -

# ## Problem settings

n = 100
data_seed = 20210103
M = 100
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
pri_beta = 0.1
rho = 0.1

### initialization
est_mu = np.random.normal(size = M)
est_ln_sigma = np.random.normal(size = M)
est_sigma = np.exp(est_ln_sigma)
est_pri_beta = pri_beta


def calc_energy(post_mu: np.ndarray, post_ln_sigma: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> float:
    post_sigma = np.exp(post_ln_sigma)
    n, M = X.shape
    energy = 0
    energy += ((y-X@post_mu)**2).sum()/2 + (X**2).sum(axis=0)@post_sigma**2/2 + n/2*np.log(2*np.pi) - M - M*np.log(pri_beta)
    energy += (-np.log(post_sigma) + np.sqrt(2)/pri_beta*np.abs(post_mu) + post_sigma/pri_beta*np.exp(-post_sigma/np.sqrt(2)*np.abs(post_mu))).sum()    
    return energy


def calc_energy_dash(post_mu: np.ndarray, post_ln_sigma: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> float:
    post_sigma = np.exp(post_ln_sigma)
    n, M = X.shape
    energy = 0
    energy += ((y-X@post_mu)**2).sum()/2 + (X**2).sum(axis=0)@post_sigma**2/2 + n/2*np.log(2*np.pi) - n*M - n*M*np.log(pri_beta)
    energy += n*(-np.log(post_sigma) + np.sqrt(2)/pri_beta*np.abs(post_mu) + post_sigma/pri_beta*np.exp(-post_sigma/np.sqrt(2)*np.abs(post_mu))).sum()    
    return energy


def calc_energy_wrapper(est_params: np.ndarray, X:np.ndarray, y:np.ndarray, pri_beta: float) -> np.ndarray:
    post_mu = est_params[:M]
    post_ln_sigma = est_params[M:]
    return calc_energy(post_mu, post_ln_sigma, X, y, pri_beta)
    pass


def df_param(X :np.ndarray, y :np.ndarray, mu: np.ndarray, sigma: np.ndarray, pri_beta: float):
    pdf_mu = np.exp(-sigma/np.sqrt(2)*np.abs(mu))*sigma/pri_beta
    dFdm = -X.T @ (y - X @ mu) + np.sqrt(2)/pri_beta*np.sign(mu) - pdf_mu*sigma/np.sqrt(2)*np.sign(mu) 
    dFds = (X**2).sum(axis = 0)*sigma - 1/sigma + pdf_mu*(1-sigma/np.sqrt(2)*np.abs(mu))
    
    return dFdm, dFds


def df_param_dash(X :np.ndarray, y :np.ndarray, mu: np.ndarray, sigma: np.ndarray, pri_beta: float):
    n = len(y)
    pdf_mu = np.exp(-sigma/np.sqrt(2)*np.abs(mu))*sigma/pri_beta
    dFdm = -X.T @ (y - X @ mu) + n*np.sqrt(2)/pri_beta*np.sign(mu) - n*pdf_mu*sigma/np.sqrt(2)*np.sign(mu) 
    dFds = (X**2).sum(axis = 0)*sigma - n/sigma + n*pdf_mu*(1-sigma/np.sqrt(2)*np.abs(mu))
    
    return dFdm, dFds


# +
train_X = np.random.normal(size = (n, M))
train_Y = train_X @ true_w + np.random.normal(size = n)

test_X = np.random.normal(size = (n, M))
test_Y = test_X @ true_w + np.random.normal(size = n)
# -

for ite in range(iteration):
    res = minimize(
        fun=calc_energy_wrapper, x0=np.hstack([est_mu, est_ln_sigma]), 
        args=(train_X, train_Y, est_pri_beta), method = "L-BFGS-B", options={"disp":True, "maxiter": 1}
    )

    est_mu = res.x[:M]
    est_ln_sigma = res.x[M:]
    est_sigma = np.exp(est_ln_sigma)
    est_pri_beta = (np.sqrt(2)*np.abs(est_mu) + est_sigma*np.exp(-est_sigma/np.sqrt(2)*np.abs(est_mu))).mean()

    dFdm, dFds = df_param(train_X, train_Y, est_mu, est_sigma, est_pri_beta)
    
    print(res.fun, (dFdm**2).mean(), (dFds**2).mean())

post_mu = res.x[:M]
post_sigma = np.exp(res.x[M:])

clf = LassoCV(fit_intercept=False)
clf.fit(train_X, train_Y)

print(np.sqrt(((test_Y - test_X@clf.coef_)**2).mean()))
print(np.sqrt(((test_Y - test_X@post_mu)**2).mean()))
print(np.sqrt(((test_Y - test_X@true_w)**2).mean()))

dFdm = -train_X.T @ (train_Y - train_X @ est_mu) + 1/pri_beta * np.sign(est_mu) * (1 - np.exp(-np.abs(est_mu) / est_sigma))
dFds = -1/est_sigma + (train_X**2).sum(axis = 0) + 1/pri_beta * (1 + np.abs(est_mu)/est_sigma) * np.exp(-np.abs(est_mu) / est_sigma)

est_mu += rho * (dFdm * est_sigma **2)
est_sigma += rho * (dFds * est_sigma **2)

print((dFdm**2).sum(), (dFds**2).sum())

est_mu + 0.1 * (dFds * est_sigma**2)

est_sigma + 0.1 * (dFds * est_sigma**2)

est_mu

dFds * est_sigma**2
