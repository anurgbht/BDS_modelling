import os
import numpy as np
import pandas as pd
import BDS_module_13_10_17 as bds
import random
import pylab as plt
from tqdm import tqdm
import time

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/1_data')
all_files = os.listdir()
##all_files = all_files[:3]
count = 0
all_start = time.time()
for file_name in all_files:
    start = time.time()
    count += 1
    print(count)
    file = pd.read_csv(file_name)
    file = file.iloc[170:260,:]
    file = file.reset_index().drop('index',1)
    mw = file.loc[0,'M.width']
    active_list = bds.find_active(mw)
    active_list = [int(file_name.split('_')[4])+1]
    ml  = file.loc[:,'M.level']
    cs = file.loc[:,'C.speed']
    feat_temp = []
    for i in range(10,file.shape[0]):
        if i<60:
            temp = file.iloc[:i,:]
        else:
            temp = file.iloc[i-60:i,:]
        feat_temp.append(bds.make_feature_ANN(temp,active_list,i,file_name,12))
