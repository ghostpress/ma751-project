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
init_lambda = 50  # TODO: write cross-validation function

model = LassoRegression(train_X_norm, train_y_num, init_lambda)
#model.fit()

print("Ok.")
