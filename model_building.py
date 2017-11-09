import os
import numpy as np
import pandas as pd
import BDS_module_17_10_17 as mf
import random
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import time

##################################################################################################
##################################################################################################
##################################################################################################
##def get_cat(x_temp):
##    x3 = x_temp[2]
##    x5 = x_temp[4]
##    if x5 < 8.095:
##        if x3 < 32.865:
##            tt = -3.42043
##        elif x3 >= 32.865:
##            if x5 < 2.9539:
##                tt = -0.2697444
##            elif x5 >= 2.9539:
##                tt = 3.4478845
##    elif x5 >= 8.095:
##        if x3 < 10.795:
##            tt = -0.1897017
##        elif x3 >= 10.795:
##            tt = 5.2923252
##    return tt

def get_cat(x_temp):
    x2 = x_temp[2]
    x4 = x_temp[4]
    if x4 < 12.45:
        tt = -5.31188811
    elif x4 >= 12.45:
        if x2 < 5.445:
            tt = -0.6965857
        elif x2 >= 5.445:
            tt = 5.8407746
    return tt
##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data')

dat_1 = pd.read_csv('test_dat_29_09_17.csv')
X_train = dat_1.iloc[:,:-1]
Y_train = dat_1.iloc[:,-1]

dat_0 = pd.read_csv('test_dat_zero_29_09_17.csv')
all_index = range(dat_0.shape[0])
test_index = random.sample(all_index,4500)
validation_index = [x for x in all_index if x in test_index]
dat_0_test = dat_0.loc[test_index,:].reset_index().drop('index',1)
dat_0_validate = dat_0.loc[validation_index,:].reset_index().drop('index',1)
X_test = dat_0_test.iloc[:,:-1]
Y_test = dat_0_test.iloc[:,-1]
print(X_test.shape,X_train.shape)

X_all = pd.concat([X_train,X_test],axis=0).reset_index().drop('index',1)
file_names = X_all.iloc[:,0]
X_all = X_all.iloc[:,1:]
##X_all = X_all.drop(['7'],axis=1)
##temp = X_all.apply(get_cat,axis=1)
##X_all = pd.concat([X_all,pd.DataFrame(temp)],axis=1)
Y_all = pd.concat([Y_train,Y_test],axis=0).reset_index().drop('index',1)
print(X_all.shape,' : X_all shape')
clf_logit,miss_files,out_files = mf.prelim_logit(X_all,X_all,Y_all,Y_all,file_names,0.2,1)
print(clf_logit.intercept_,clf_logit.coef_)
print(miss_files)

##clf_gbm = mf.prelim_gbm(X_all,Y_all,0.2)
##clf_keras = mf.prelim_keras(X_all,Y_all,0.2)

##dat_all = pd.concat([file_names,X_all,Y_all,out_files],axis=1)
##dat_all.to_csv('data_for_model.csv',index=False)



##coef_list = []
##inter_list = []
##
##for i in range(50):
##    start = time.time()
##    test_index = random.sample(all_index,5000)
##    validation_index = [x for x in all_index if x in test_index]
##    dat_0_test = dat_0.loc[test_index,:].reset_index().drop('index',1)
##    dat_0_validate = dat_0.loc[validation_index,:].reset_index().drop('index',1)
##    X_test = dat_0_test.iloc[:,:-1]
##    Y_test = dat_0_test.iloc[:,-1]
##
####    X_validate = dat_0_validate.iloc[:,:-1]
####    X_validate = X_validate.drop(['0','5','8','10','11','12','16','18','22'],axis=1)
####    Y_validate = dat_0_validate.iloc[:,-1]
##
##    X_all = pd.concat([X_train,X_test],axis=0).reset_index().drop('index',1)
##    X_all = X_all.drop(['3','4','8','10','11','12','13','14','16','17','18','22'],axis=1)
##    Y_all = pd.concat([Y_train,Y_test],axis=0).reset_index().drop('index',1)
##
##    X_train, X_test, y_train, y_test,test_index = mf.train_test_split_my(X_all, Y_all, test_size=0.3)
##    clf = LogisticRegression( max_iter=10000,C=0.5,verbose=0)
##    clf.fit(X_train,y_train)
##    coef_list.append(clf.coef_)
##    inter_list.append(clf.intercept_)
##    print(i+1,(time.time() - start)/60)
##    
##coef_all = pd.DataFrame([x.tolist()[0] for x in coef_list])
##inter_all = [x.tolist()[0] for x in inter_list]

##    X_validate = dat_0_validate.iloc[:,:-1]
##    X_validate = X_validate.drop(['0','5','8','10','11','12','16','18','22'],axis=1)
##    Y_validate = dat_0_validate.iloc[:,-1]
