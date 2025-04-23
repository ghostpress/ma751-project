import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasso import LassoRegression
import utils

path_to_data = utils.get_data_directory()
path_to_model = utils.get_model_output_directory()

run_before = not utils.is_folder_empty(path_to_model)

if not run_before:

    # Load training data; may contain text-valued columns
    train_X = pd.read_csv(path_to_data + "/train_X.csv", index_col=0)
    train_y = pd.read_csv(path_to_data + "/train_y.csv", index_col=0)

    # Convert to numeric-only
    train_X_num = utils.numeric_only(train_X)
    train_y_num = train_y.to_numpy().flatten()
    # TODO: use Firas' variable selection/best subset analysis to select training data

    # Normalize X, to ease GD convergence
    train_X_norm = utils.normalize(train_X_num)

    # Potential regularization parameter values, to select with cross-validation
    lambda_vals = [0.0, 0.001, 0.1, 1.0, 10.0, 100.0]

    model = LassoRegression.new_model(train_X_norm, train_y_num)
    model.fit(cross_val=True, nfolds=10, param_vals=lambda_vals, iterations=10000)  
    model.save(utils.get_model_output_directory())  

else:

    # Load validation data
    val_X = pd.read_csv(path_to_data + "/val_X.csv", index_col=0)
    val_y = pd.read_csv(path_to_data + "/val_y.csv", index_col=0)

    val_X_num = utils.numeric_only(val_X)
    val_y_num = val_y.to_numpy().flatten()

    # Load fitted model parameters
    fitted_B = np.load(path_to_model + "/beta.npy")
    fitted_B0 = np.load(path_to_model + "beta0.npy")
    fitted_lambda = np.load(path_to_model + "lambda.npy")

    model = LassoRegression.from_params(val_X_num, val_y_num, fitted_lambda, fitted_B0, fitted_B)

    # TODO: create a self.errors object to track MSE per iteration/fold and plot in run.py
    # TODO: check Hastie 7.10.2 and do diagnostic plots to show cross-validation done "the right way"

    # Plot model-fit results
    yhat = model.predict(val_X_num)
    #print(yhat)
    #print(val_y_num)

    plt.figure(figsize=(10,10))
    plt.scatter(val_y_num, yhat, c='crimson')
    #plt.yscale('log')
    #plt.xscale('log')
    #p1 = max(max(yhat), max(yhat))
    #p2 = min(min(yhat), min(yhat))
    #plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()


print("Ok.") 
