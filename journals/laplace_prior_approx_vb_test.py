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

# # Implementation to estimate linear regression with Laplace by approximated Laplace prior
# + We consider the Gaussian approximated Laplace prior:
# $$
# |w_j| \approx |\xi_j| + \frac{1}{2\sqrt{\xi_j}}(w_j^2 - \xi_j^2).
# $$  
# This is derived from condition of $C^1$ convex function because $|w_j| = \sqrt{w_j^2}$ and $\sqrt{x}$ is convex function.
# + Note that $|\xi_j|$ is optimized by objective function.

# %load_ext autoreload
# %autoreload 2

import numpy as np
from scipy.stats import wishart

# data setting
n = 20
M = 40
zero_features = M // 2
data_seed = 20201010

# problem setting
np.random.seed(data_seed)
true_w = np.random.normal(scale = 3, size = M)
zero_ind = np.random.choice(M, size = zero_features, replace = False)
true_w[zero_ind] = 0

# data generation
train_X = np.random.normal(size = (n, M))
train_Y = np.random.normal(size = n) + train_X @ true_w

# learning setting
seed = 20201011
pri_beta = 0.001
iteration = 1000
tol = 1e-8
is_trace = True
step = 10
opt_pri_beta = True


def _obj_func(y, pri_beta, mean, inv_sigma, h_xi, v_xi):
    F = 0
    F += pri_beta/2*np.sqrt(h_xi).sum() + v_xi@h_xi - M*np.log(pri_beta/2)
    F += n/2*np.log(2*np.pi) + train_Y@train_Y/2 - mean @ (inv_sigma @ mean)/2 + np.linalg.slogdet(inv_sigma)[0]/2
    return F


# initialization
if seed > 0:
    np.random.seed(seed)
    pass
mean = np.random.normal(size = M)
inv_sigma = wishart.rvs(df = M + 2, scale = np.eye(M), size = 1)
sigma = np.linalg.inv(inv_sigma)
est_pri_beta = pri_beta if not opt_pri_beta else np.random.gamma(shape = 3, size = 1)

F = []
for ite in range(iteration):
    # update form of approximated laplace prior
    est_h_xi = mean**2 + np.diag(sigma)
    est_v_xi = -pri_beta/2/np.sqrt(est_h_xi)

    # update posterior distribution
    inv_sigma = train_X.T @ train_X -2*np.diag(est_v_xi)
    mean = np.linalg.solve(inv_sigma, train_X.T @ train_Y)
    sigma = np.linalg.inv(inv_sigma)

    est_pri_beta = M/((mean**2 + np.diag(sigma))/(2*np.sqrt(est_h_xi))).sum()
    
    current_F = _obj_func(train_Y, est_pri_beta, mean, inv_sigma, est_h_xi, est_v_xi)
    F.append(current_F)
    if ite > 0 and np.abs(current_F - F[ite-1]) < tol:
        break
    if is_trace and ite % step == 0:
        print(current_F)


