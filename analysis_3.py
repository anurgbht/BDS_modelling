import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
import my_functions as mf

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat2')
all_files = os.listdir()

temp=pd.DataFrame()
count = 0
X_all = pd.DataFrame()
Y_all = pd.DataFrame()
index_file = pd.DataFrame()
for file_name in all_files:
    count = count + 1
    print(count)
    file = pd.read_csv(file_name)
    if file.loc[0,'layer'] == 2 :
        TC_layer = int(np.mean(file.loc[:,'TC_layer']))
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        ML = file.loc[:,'M.level']
        CS = file.loc[:,'C.speed']
        CP = file.loc[:,'C.percent']
        MW = file.loc[:,'M.width']

        TC_layer_left = find_left(TC_layer,file)

        tt1 = 'TC' + str(TC_layer_left)
        tt2 = 'TC' + str(TC_layer_left + 20)

        LL1 = file.loc[:,tt1]
        LL2 = file.loc[:,tt2]

        TC_layer_right = find_right(TC_layer,file)

        tt1 = 'TC' + str(TC_layer_right)
        tt2 = 'TC' + str(TC_layer_right + 20)

        LR1 = file.loc[:,tt1]
        LR2 = file.loc[:,tt2]

        TC_layer_opp = find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]

        tx,ty,index_temp = make_x(L1,L2,ML,CP,CS,LL1,LL2,LR1,LR2,LO1,LO2,MW,file_name.split('_')[6],file_name)
        index_file = index_file.append(index_temp)
        X_all = X_all.append(tx)
        Y_all = Y_all.append(ty)

print(X_all.shape,' : Size of all X')
print(Y_all.shape,' : Size of all Y')

outdata = X_all
outdata['y_label'] = Y_all
outdata = outdata.dropna()
##outdata.to_csv('all_data.csv')

X_all = outdata.iloc[:,:-1]
##X_all = preprocessing.scale(X_all)
Y_all = pd.DataFrame(outdata.iloc[:,-1])

##IV_table = find_IV(X_all,Y_all,2)

##clf_RF = prelim_RF(X_all,Y_all)
##clf_logit = prelim_logit(X_all,Y_all)
clf_GBM,y_test,my_pred,test_index = prelim_GBM(X_all,Y_all,0.2)

##for i in range(0,len(my_pred)):
##    if (my_pred[i] != y_test.iloc[i,0]) & (y_test.iloc[i,0] == 1):
##        print(index_file.iloc[test_index.iloc[i,0],0], ' with Original tag : ', y_test.iloc[i,0])
####        os.remove(index_file.iloc[test_index.iloc[i,0],0])
##        print("File removed")
