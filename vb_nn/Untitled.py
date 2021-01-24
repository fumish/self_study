# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Relu mean test
# + Here, we calculate expected value of relu following a normal distribution:  
#     $E[relu(x)] = \int relu(x) N(x|\mu, \sigma) dx$

import numpy as np

np.random.seed(20201012)

mu = 2
sigma = 2
X = np.random.normal(loc = mu, scale = sigma, size = 1000)

relu = lambda x: np.clip(x, 0, None)
relu(X)


