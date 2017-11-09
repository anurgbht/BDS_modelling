import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
from time import sleep
import my_functions as mf

########################################################################################
#################################~MAIN~################################################
########################################################################################

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat')
all_files = os.listdir()
ttt=(len(all_files))
temp=pd.DataFrame()
count = 0
X_all_trigger = pd.DataFrame()
X_all_left = pd.DataFrame()
X_all_right = pd.DataFrame()
Y_all = pd.DataFrame()
index_file = pd.DataFrame()
for file_name in all_files:
    count = count + 1
    print(count,"/",ttt, file_name)
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

        TC_layer_left = mf.find_left(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_left)
        tt2 = 'TC' + str(TC_layer_left + 20)

        LL1 = file.loc[:,tt1]
        LL2 = file.loc[:,tt2]

        TC_layer_right = mf.find_right(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_right)
        tt2 = 'TC' + str(TC_layer_right + 20)

        LR1 = file.loc[:,tt1]
        LR2 = file.loc[:,tt2]

        TC_layer_opp = mf.find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]
        tx,ty,index_temp = mf.make_x(L1,L2,ML,CP,CS,LO1,LO2,MW,file_name.split('_')[6],file_name)
        X_all_trigger = X_all_trigger.append(tx)

        tx,ty,index_temp = mf.make_x(LL1,LL2,ML,CP,CS,LO1,LO2,MW,file_name.split('_')[6],file_name)
        X_all_left = X_all_left.append(tx)

        tx,ty,index_temp = mf.make_x(LR1,LR2,ML,CP,CS,LO1,LO2,MW,file_name.split('_')[6],file_name)
        X_all_right = X_all_right.append(tx)

        Y_all = Y_all.append(ty)
        index_file = index_file.append(index_temp)

temp = pd.concat([X_all_left,X_all_trigger,X_all_right,Y_all,index_file],axis=1)
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

temp.to_csv('data_dump_24_3_17.csv',index=False)


X_all_left = pd.DataFrame(pp.scale(temp.iloc[:,0:20]))
X_all_trigger = pd.DataFrame(pp.scale(temp.iloc[:,20:40]))
X_all_right = pd.DataFrame(pp.scale(temp.iloc[:,40:60]))
Y_all_clean = temp.iloc[:,60]
index_file_clean = temp.iloc[:,61]

print(X_all_trigger.shape,' : Size of all X trigger')
print(X_all_left.shape,' : Size of all X left')
print(X_all_right.shape,' : Size of all X right')
print(Y_all_clean.shape,' : Size of all Y')
print(index_file_clean.shape,' : Shape of index file')



##clf_RF = prelim_RF(X_all,Y_all)
##clf_logit = prelim_logit(X_all,Y_all)
##clf_GBM_trigger,y_test,my_pred,test_index = prelim_GBM(X_all_trigger,Y_all_clean,0.2)
##clf_GBM_left,y_test,my_pred,test_index = prelim_GBM(X_all_left,Y_all_clean,0.2)
##clf_GBM_right,y_test,my_pred,test_index = prelim_GBM(X_all_right,Y_all_clean,0.2)
##
##my_pred_trigger = my_pred_fun(clf_GBM_trigger,X_all_trigger,Y_all_clean,0.2)
##print(confusion_matrix(Y_all_clean,my_pred_trigger))
##my_pred_left = my_pred_fun(clf_GBM_left,X_all_left,Y_all_clean,0.2)
##print(confusion_matrix(Y_all_clean,my_pred_left))
##my_pred_right = my_pred_fun(clf_GBM_right,X_all_right,Y_all_clean,0.2)
##print(confusion_matrix(Y_all_clean,my_pred_right))
##
##my_pred_trigger = clf_GBM_trigger.predict_proba(X_all_trigger)
##my_pred_left = clf_GBM_left.predict_proba(X_all_left)
##my_pred_right = clf_GBM_right.predict_proba(X_all_right)

##my_pred_trigger.to_csv('my_pred_trigger.csv')
##my_pred_left.to_csv('my_pred_left.csv')
##my_pred_right.to_csv('my_pred_right.csv')
##Y_all_clean.to_csv('Y_all.csv')

##index_false = pd.DataFrame()
##
##for i in range(len(my_pred)):
##    if (Y_all_clean.iloc[i][0] != my_pred.iloc[i]) & Y_all_clean.iloc[i][0] == 1:
##        index_false = index_false.append([i])
##
##misclassified_files = [index_file.iloc[x][0] for x in index_false.iloc[:,0]]
##print(misclassified_files)
