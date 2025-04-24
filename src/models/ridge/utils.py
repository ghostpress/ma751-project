import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error


def get_data_directory():
    cwd = os.getcwd()
    
    with open(cwd + "/config.yml", "r") as file: 
        config_settings = yaml.safe_load(file)
    
    path_to_data = config_settings.get("dataparent")
    return path_to_data


def get_model_output_directory():
    cwd = os.getcwd()

    with open(cwd + "/config.yml", "r") as file: 
        config_settings = yaml.safe_load(file)
    
    path_to_model = config_settings.get("modelparent")
    return path_to_model


def get_fig_output_directory():
    cwd = os.getcwd()

    with open(cwd + "/config.yml", "r") as file: 
        config_settings = yaml.safe_load(file)
    
    path_to_figs = config_settings.get("figsparent")
    return path_to_figs


def numeric_only(data):
    droplist = []

    for col in data.columns:
        if not is_numeric_dtype(data[col]):
            droplist.append(col)

    num = data.drop(droplist, axis=1, inplace=False)
    return num.to_numpy()


def normalize(X):
    mean = np.mean(X)
    sd = np.std(X)

    newX = (X - mean) / sd
    return newX


def nonzero(X):
    droplist = []

    for col in range(X.shape[1]):
        if np.all(X[:,col] == 0):
            droplist.append(col)

    nonzero = np.delete(X, droplist, axis=1)
    return nonzero


def is_folder_empty(parent):
    contents = os.listdir(parent)
    return (len(contents) == 0)
    

def simulate_data():
    X = 10*np.random.random((15,5))

    # Add collinear columns
    col1 = 5*X[:,3]
    col2 = X[:,1] - 100
    col3 = np.exp(X[:,-1])
    
    X = np.c_[X, col1, col2, col3] 

    thetas = np.random.random(X.shape[1]) 
    y = X.dot(thetas)

    y += np.random.normal(loc=0, scale=.1, size=y.shape)
    return X, y


def plot_prediction(y1, y2, title="Model Evaluation"):        
    plt.figure(figsize=(8,8))
    plt.title(title, fontsize=17)
    plt.ylabel("Predicted SMB", fontsize=16)
    plt.xlabel("Observed SMB", fontsize=16)
    plt.scatter(y1, y2)
    plt.tick_params(labelsize=10)
    lineStart = -2.5
    lineEnd = 1.5
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((r'$RMSE=%.2f$' % (mean_squared_error(y1, y2),), r'$R^2=%.2f$' % (r2_score(y1, y2), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    print(f"Saving plot of {title} to {get_fig_output_directory()}")
    filename = title.replace(" ", "-") + ".png"
    plt.savefig(get_fig_output_directory() + "/" + filename)
    pass

