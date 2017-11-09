import os
import numpy as np
import pandas as pd
import BDS_module_17_10_17 as bds
import random
import pylab as plt
from tqdm import tqdm
import time

##################################################################################################
##################################################################################################
##################################################################################################
def make_all(path_read,path_save,tag):
    os.chdir(path_read)
    all_files = os.listdir()
    ##all_files = all_files[:3]
    count = 0
    all_start = time.time()
    for file_name in all_files:
        start = time.time()
        count += 1
        print(count,len(all_files))
        file = pd.read_csv(file_name)
        mw = file.loc[0,'M.width']
        active_list = bds.find_active(mw)
        ml  = file.loc[:,'M.level']
        cs = file.loc[:,'C.speed']
        prob_temp = []
        for i in range(10,file.shape[0]):
            if i<60:
                temp = file.iloc[:i,:]
            else:
                temp = file.iloc[i-60:i,:]
            prob_temp.append(bds.prob_level_three(temp,active_list,i,file_name,tag))
        prob_temp = pd.DataFrame(prob_temp)
    ##    print(prob_temp)
        prob_temp.to_csv(path_save+file_name.split('.')[0]+'_level_three.csv',index=False)

##################################################################################################
##################################################################################################
##################################################################################################

path_read = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/first iteration/caster C old their alarms'
path_save = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/caster C old pred 23 10/'
make_all(path_read,path_save,12)

path_save = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/caster C old pred 23 10 23/'
make_all(path_read,path_save,23)
##
##path_read = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/caster B recent'
##path_save = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/caster B recent pred 18 10/'
##make_all(path_read,path_save,12)
##
##path_save = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/caster B recent pred 18 10 23/'
##make_all(path_read,path_save,23)
##
