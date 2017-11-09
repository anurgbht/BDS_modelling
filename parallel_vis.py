import multiprocessing as mp
import time
import numpy as np
import random
import os
import pandas as pd
import pickle

###############################################################################
###############################################################################
###############################################################################

os.chdir("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/pickle_test")

for pickle_file in os.listdir():
    temp = []
    with open(pickle_file,'rb') as f:
        results = pickle.load(f)
        for i in range(len(results)):
            for j in range(len(results[0])):
                temp.append([x[0][1] for x in results[i][j]])
    temp = pd.DataFrame(temp)
    temp = temp.transpose()
    print(temp.shape)
    temp.to_csv(pickle_file.split('.')[0] + '_predicted.csv')
