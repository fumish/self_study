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

# # L0 greedy test
# $$
#  minimize_w \|w\|_0 \ st. \ y = Xw
# $$
# To solve, first we consider about greedy method.

# # Preliminary Section

# ## Import library

# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

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

### mutual coherence
mutual_coherence = lambda x: np.triu(cosine_similarity(x.T),k=1).max()

sq_l2_train_X = (train_X**2).sum(axis = 0)
norm_train_X = train_X / np.sqrt(sq_l2_train_X)

est_w = np.zeros(M)

i1 = np.argmax((train_Y @ norm_train_X)**2)
est_w[i1] = (train_X[:,i1] @ train_Y) / sq_l2_train_X[i1]
residual = train_Y - train_X @ est_w

i2 = np.argsort(-(residual @ norm_train_X)**2)[1]
est_w[i2] = (train_X[:,i2] @ train_Y) / sq_l2_train_X[i2]
residual = train_Y - train_X @ est_w

train_X[:,i2] @ residual

true_w

true_w[i2]

est_w

w1
