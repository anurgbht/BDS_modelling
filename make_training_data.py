import os
import numpy as np
import pandas as pd
import implement_functions as mf
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/training data/0_data')
all_files = os.listdir()
##all_files = all_files[:5]
temp=pd.DataFrame()
count = 0
X_all = []
Y_all = []
for file_name in all_files:
    count = count + 1
    print(count,len(all_files))
    file = pd.read_csv(file_name)
    TC_active = mf.find_active(file.loc[0,'M.width'])
    for TC_layer in TC_active:
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)
        
        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        ML = file.loc[:,'M.level']
        CS = file.loc[:,'C.speed']
        CP = file.loc[:,'C.percent']

        tx,ty = mf.make_x_3600(L1,L2,ML,CP,CS)
        X_all.extend(tx)
        Y_all.extend(ty)
X_all = pd.DataFrame(X_all)
Y_all = pd.DataFrame(Y_all)


##X_all = pd.read_csv('x_data.csv')
##Y_all = pd.read_csv('y_data.csv')
##
####clf_gbm = mf.prelim_gbm(X_all,Y_all,0.9)
##clf_logit = mf.prelim_logit(X_all,Y_all,0.3,1)
