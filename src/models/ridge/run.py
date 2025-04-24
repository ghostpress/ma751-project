import numpy as np
import pandas as pd
from ridge import RidgeRegression
import utils

path_to_data = utils.get_data_directory()
path_to_model = utils.get_model_output_directory()
path_to_figs = utils.get_fig_output_directory()

run_before = not utils.is_folder_empty(path_to_model)
np.set_printoptions(precision=3)

if not run_before:

    # Load training data; may contain text-valued columns
    train_X_raw = pd.read_csv(path_to_data + "/train_X.csv", index_col=0)
    train_y_raw = pd.read_csv(path_to_data + "/train_y.csv", index_col=0)

    # Convert to numeric-only
    train_X_num = utils.numeric_only(train_X_raw)
    train_y = train_y_raw.to_numpy().flatten()

    # Ensure X has no all-zero columns (will need to invert it)
    train_X_nonz = utils.nonzero(train_X_num)

    # Add a column of ones to X
    train_X = np.c_[np.ones(train_X_nonz.shape[0]), train_X_nonz]  

    # Fit the linear parameters
    model = RidgeRegression.new_model(train_X, train_y)
    model.fit()

    # Potential regularization parameter values, to select with cross-validation
    lambda_vals = [0.0, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
  
    # Now fit lambda regularization term
    model.fit(cross_val=True, nfolds=10, param_vals=lambda_vals)
    model.save(utils.get_model_output_directory())  

    yhat = model.predict(train_X)
    utils.plot_prediction(train_y, yhat)

else:
    # Load validation data
    val_X = pd.read_csv(path_to_data + "/val_X.csv", index_col=0)
    val_y = pd.read_csv(path_to_data + "/val_y.csv", index_col=0)

    val_X_num = utils.numeric_only(val_X)

    # Add a column of ones to X
    val_X = np.c_[np.ones(val_X_num.shape[0]), val_X_num]
    val_X = utils.nonzero(val_X)
    val_y = val_y.to_numpy().flatten()

    # Load fitted model parameters
    fitted_B = np.load(path_to_model + "/beta.npy")
    fitted_lambda = np.load(path_to_model + "lambda.npy")

    model = RidgeRegression.from_params(val_X, val_y, fitted_lambda, fitted_B)

    # Plot model-fit results
    yhat = model.predict(val_X)
    utils.plot_prediction(val_y, yhat, title="Ridge Regression: Observed vs Predicted (unseen data)")


print("Ok.") 
