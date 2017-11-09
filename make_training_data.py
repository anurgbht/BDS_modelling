import os
import numpy as np
import pandas as pd
import BDS_module_12_10_17 as bds
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
from tqdm import *

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/1_data')
all_files = os.listdir()
##all_files = ["F5240226_092_01_2_02_CASTER-C_True_May'15.csv"]
##all_files = all_files[:4]
count = 0
X_all = []
Y_all = []
for file_name in all_files:
    count = count + 1
    print(100*(count/len(all_files)),count,len(all_files))
    file = pd.read_csv(file_name)
    TC_layer = int(file_name.split('_')[4].split('.')[0])+1
    tt1 = 'TC' + str(TC_layer)
    tt2 = 'TC' + str(TC_layer + 20)
    
    L1 = file.loc[:,tt1]
    L2 = file.loc[:,tt2]
    CS = file.loc[:,'C.speed']
    
    tx,ty = bds.make_x_300_stay(L1,L2,CS,file_name)
    if len(tx)>0:
        X_all.extend(tx)
        Y_all.extend(ty)
X_all = pd.DataFrame(X_all)
Y_all = pd.DataFrame(Y_all)
dat_all = pd.concat([X_all,Y_all],axis=1)
##print(dat_all.transpose())
input('Exit ?')
dat_all.to_csv('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/'+'test_dat_12_10_17.csv',index=False)
##
##################################################################################################
##################################################################################################
os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/0_data_test')
all_files = os.listdir()
##all_files = ["H2200453_100_01_4_11.csv"]
##random.shuffle(all_files)
##all_files = all_files[:5]
count = 0
X_all = []
Y_all = []
for file_name in all_files:
    count = count + 1
    print(100*(count/len(all_files)),count,len(all_files))
    file = pd.read_csv(file_name)
    CS = file.loc[:,'C.speed']
    TC_active = bds.find_active(file.loc[0,'M.width'])
##    TC_active = [12]
    for TC_layer in TC_active:
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)            
        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        tx,ty = bds.make_x_online(L1,L2,CS,TC_layer,file_name)
        X_all.extend(tx)
        Y_all.extend(ty)
X_all = pd.DataFrame(X_all)
print(X_all.shape,' : size of the data set')
input('Exit ?')
Y_all = pd.DataFrame(Y_all)
dat_all = pd.concat([X_all,Y_all],axis=1)
dat_all.to_csv('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/'+'test_dat_zero_12_10_17.csv',index=False)
