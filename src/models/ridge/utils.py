import os
import yaml
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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


def is_folder_empty(parent):
    contents = os.listdir(parent)
    return (len(contents) == 0)
    
