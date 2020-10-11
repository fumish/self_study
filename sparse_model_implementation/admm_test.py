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

# # Test for Alternative Direction Method of Multipliers
# + Object function $L(w)$
# $$
# L(w) = \frac{1}{2} \|Y - Xw\|_2^2 + \lambda \|Aw\|_1,
# $$
# where $w \in \mathbb{R}^M$, $X \in \mathbb{R}^{n \times M}$, $Y \in \mathbb{R}^n$ , and $A \in PSD(M)$ is hyperparameter.
# Here we want to optimize the parameter $w$.  
# Moreover, parameter w has zero elements with a zero_ratio.

# # Preliminary Section

# ## Import library

# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt


# ## Basic functions

def soft_thresholing(x:np.ndarray, gamma:float):
    """
    soft thresholding function S_gamma(x):
    x >= gamma:
        S_gamma(x) = x - gamma
    |x| < gamma:
        S_gamma(x) = 0
    x <= -gamma:
        S_gamma(x) = x + gamma
        
    + Input:
        1. x: M-dim array
        2. gamma: positive real value
    """
    
    pos_dom_ind = np.where(x >= gamma)[0]
    neg_dom_ind = np.where(x <= -gamma)[0]
    
    ret_val = np.zeros(len(x))
    ret_val[pos_dom_ind] = x[pos_dom_ind] - gamma
    ret_val[neg_dom_ind] = x[neg_dom_ind] + gamma
    
    return ret_val


# ## Problem settings

# +
n = 200
M = 80
data_seed = 20200726

zero_ratio = 0.8
# -

# ## Data Generation
# $$
#  y_i \sim x_i w^* \forall i,
# $$
# $w^* \sim N(0,I_M)$, and some elements are zero. $x_i \sim N(0,I_M)$
#

np.random.seed(data_seed)
train_X = np.random.normal(size = (n, M))
true_w = np.random.normal(size = M)
### some elements are zero.
true_w[np.random.choice(np.arange(M), size = int(zero_ratio * M), replace = False)] = 0
train_Y = np.random.normal(train_X @ true_w, size = n)

# ## Learning settings

params = {
    "iteration": 2000,
    "lam": 10,
    "gamma": 1,
    "A": -1*np.eye(M) + (np.triu(np.ones((M, M)), k=1) - np.triu(np.ones((M, M)), k=2)),
    ##"A": np.eye(M) + (np.tril(np.ones((M, M)), k=-1) - np.tril(np.ones((M, M)), k=-2))*0.5 + (np.triu(np.ones((M, M)), k=1) - np.triu(np.ones((M, M)), k=2))*0.5,
    "seed": 20190726,
    "eps": 1e-5
}

# # Learning

### unpack for functionize
gamma = params["gamma"]
lam = params["lam"]
seed = params["seed"]
iteration = params["iteration"]
A = params["A"]
eps = params["eps"]

np.random.seed(seed)

### initialization
est_w = np.random.normal(size = M)
est_z = np.random.normal(size = M)
est_v = np.random.normal(size = M)
# est_v = np.zeros(M)

### put fixed values
inv_S = np.linalg.inv(train_X.T @ train_X + 1/gamma * A.T @ A)
loss_func = lambda x,y,w,A,lam: ((x@w - y)**2).sum()/2 + lam*(np.abs(A@w)).sum()
generalized_lagrangian = lambda x,y,w,z,v,A,lam,gamma: ((x@w - y)**2).sum()/2 + lam*np.abs(z).sum()+v@(A@w - z)/gamma+((A@w - z)**2).sum()/(2*gamma)

loss_func_trace = np.zeros(iteration)
lagrangian_trace = np.zeros(iteration)
for ite in range(iteration):
    est_w = inv_S @ (train_X.T @ train_Y + 1/gamma * A.T @ (est_z - est_v))
    est_z = soft_thresholing(A @ est_w + est_v, gamma * lam)
    est_v = est_v + A @ est_w - est_z
    
    loss_func_trace[ite] = loss_func(train_X, train_Y, est_w, A, lam)
    lagrangian_trace[ite] = generalized_lagrangian(train_X, train_Y, est_w, est_z, est_v, A, lam, gamma)
    if ite > 0 and np.abs(loss_func_trace[ite] - loss_func_trace[ite-1]) < eps:
        loss_func_trace = loss_func_trace[:(ite+1)]
        lagrangian_trace = lagrangian_trace[:(ite+1)]
        break

plt.plot(loss_func_trace)




