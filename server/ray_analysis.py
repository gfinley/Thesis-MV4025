import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import time

#function opens the directory and gets all folders that match the patter
def get_folders(path, pattern):
    folders = []
    for folder in os.listdir(path):
        if pattern in folder:
            folders.append(folder)
    return folders

exp_folders = get_folders("../", "3_IMPALA_parallel_gpu")
print(exp_folders)


path = "../" + exp_folders[0] + "/"

sub_exps = get_folders(path, "_")
print(sub_exps)

#function looks in dir for files named "progress.csv" and params.json and retuns a pd dataframe and a dict
def get_progress(path):
    for file in os.listdir(path):
        if file == "progress.csv":
            progress = pd.read_csv(path + file)
        if file == "params.json":
            params = pd.read_json(path + file)
    return progress, params




