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

import numpy as np

import sys
sys.path.append("C:\\Users\\user\\Documents\\GitHub\\LearningModels\\lib")

from util import Ga

# data settings
n = 100
M = 3 # data dimenension
data_seed = 20200920
np.random.seed(data_seed)

# data generation
K0 = 5
true_ratio = np.random.dirichlet([1]*K0)
true_mean = np.random.normal(size = (K0, M))

np.random.choice()
