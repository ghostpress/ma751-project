import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasso import LassoRegression
import utils

# Load training data; may contain text-valued columns
train_X = pd.read_csv(utils.get_data_directory() + "/train_X.csv", index_col=0)
train_y = pd.read_csv(utils.get_data_directory() + "/train_y.csv", index_col=0)

# Convert to numeric-only
train_X_num = utils.numeric_only(train_X)
train_y_num = train_y.to_numpy().flatten()

# Normalize X, to ease GD convergence
train_X_norm = utils.normalize(train_X_num)

# Potential regularization parameter values, to select with cross-validation
lambda_vals = [0.0, 0.001, 0.1, 1.0, 10.0, 100.0]

model = LassoRegression(train_X_norm, train_y_num, reg=0.0001)  # small initial lambda value
model.fit(cross_val=True, nfolds=10, param_vals=lambda_vals, iterations=100)  

# TODO: use variable selection from Firas' OLS analysis to re-train
# TODO: create a self.errors object to track MSE per iteration/fold and plot in run.py
# TODO: check Hastie's formulas for overall error after cross-validation

print("Ok.")
