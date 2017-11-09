import pandas as pd
import numpy as np
import os
import time

os.chdir("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/results")

results_file = pd.read_csv("results_all.csv")

unique_files = list(np.unique(results_file.loc[:,'file_name']))
valid_list = []
for file in unique_files:
    sample = results_file.loc[results_file.loc[:,'file_name'] == file,:]
    if (('L1-L2' in list(np.unique(sample.loc[:,'type'])))):
        print(sample)
        input('')
