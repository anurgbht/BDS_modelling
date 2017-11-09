import os
import numpy as np
import pandas as pd
import implement_functions_12_08_17 as mf
import random
import pylab as plt
from tqdm import tqdm

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/')
##all_files = os.listdir()
##all_files = all_files[:5]
all_files = ["Fc210213_097_01_2_18_CASTER-C_True_Dec'15.csv"]
count = 0

for file_name in all_files:
    count += 1
    print(count)
    file = pd.read_csv(file_name)
    mw = file.loc[0,'M.width']
    active_list = mf.find_active(mw)
    active_list = [1]
    ml  = file.loc[:,'M.level']
    cs = file.loc[:,'C.speed']
    feat_temp = []
    for i in range(10,file.shape[0]):
        if i<60:
            temp = file.iloc[:i,:]
        else:
            temp = file.iloc[i-60:i,:]
        feat_temp.extend(mf.all_features(temp,active_list))
    feat_temp = pd.DataFrame(feat_temp)
    
    feat_temp.to_csv('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test/'+file_name.split('.')[0]+'_level_two.csv',index=False)


    
