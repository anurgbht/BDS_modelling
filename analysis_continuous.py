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
import shutil
import my_functions as mf

def include_tree(row):
    x2 = row[1]
    x6 = row[5]
    x7 = row[6]
    x8 = row[7]
    x11 = row[10]
    
    if (x2<=0.324):
        if (x7<=-0.4608):
            if (x11<=-0.0885):
                if (x8 <= -0.1921):
                    tt = 'A'
                elif (x8 > -0.1921):
                    tt = 'B'
            elif(x11>-0.0885):
                tt = 'C'
        elif(x7>-0.4608):
            tt = 'D'
    elif(x2>0.324):
        if(x6<=-0.677):
            tt = 'E'
        elif(x6>-0.677):
            if(x7<=-0.2875):
                tt = 'F'
            elif(x7>-0.2875):
                tt = 'G'
    return tt

def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp

def my_logit_predict_proba(beta_0,beta,X):
    temp = pd.DataFrame()
    for i in range(X.shape[0]):
        temp2 = X.iloc[i,:]
        temp3 = 1-(1/(1+np.exp(-(np.inner(temp2,beta) + beta_0))))
        temp = temp.append([temp3])        
    return temp
    

def plot_fun_temp(logit_trigger,gbm_trigger,RF_trigger,beta_0,beta,temp_x_trigger):
    logit_trigger_my = pd.DataFrame(my_logit_predict_proba(beta_0,beta,temp_x_trigger))
    plt.plot(range(len(logit_trigger.iloc[:,-1])),logit_trigger.iloc[:,-1],
             range(len(logit_trigger.iloc[:,-1])),logit_trigger_my.iloc[:,-1],
             range(len(gbm_trigger.iloc[:,-1])),gbm_trigger.iloc[:,-1],
             range(len(gbm_trigger.iloc[:,-1])),RF_trigger.iloc[:,-1]
             )
    plt.xlabel('Time')
    plt.ylabel('Probability of being a sticker')
    plt.legend(['logit','my_logit','GBM','RF'])
    plt.grid(1)
    plt.show()

###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_C_10_3_17.csv')
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

X_all_left = pd.DataFrame(pp.scale(temp.iloc[:,0:18]))
X_all_trigger = pd.DataFrame(pp.scale(temp.iloc[:,18:36]))
X_all_right = pd.DataFrame(pp.scale(temp.iloc[:,36:54]))
Y_all_clean = temp.iloc[:,54]
index_file_clean = temp.iloc[:,55]

X_all_trigger.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18"]
X_all_left.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18"]
X_all_right.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18"]

print(X_all_trigger.shape,' : Size of all X trigger')
print(X_all_left.shape,' : Size of all X left')
print(X_all_right.shape,' : Size of all X right')
print(Y_all_clean.shape,' : Size of all Y')
print(index_file_clean.shape,' : Shape of index file')

##temp = X_all_trigger.apply(include_tree,axis=1)
##
##data_logit = pd.concat([X_all_trigger,temp],axis=1)
##
##print(X_all_trigger.shape)
##print(data_logit.shape)
##
##global woe_table
##woe_table = mf.calc_WOE(data_logit.iloc[:,-1],Y_all_clean)
##
##temp2 = data_logit.apply(subs_WOE,axis=1)
##
### just to make sure everything is going okay
##temp3 = pd.concat([temp2,data_logit.iloc[:,-1]],axis=1)
##
##data_logit = pd.concat([X_all_trigger,temp2],axis=1)
##
####clf_logit = mf.prelim_logit(data_logit,Y_all_clean,0.1,1)



clf_logit_trigger = mf.prelim_logit(X_all_trigger,Y_all_clean,0.1,0)
beta_0 = clf_logit_trigger.intercept_
beta = clf_logit_trigger.coef_
clf_gbm_trigger = mf.prelim_gbm(X_all_trigger,Y_all_clean,0.1)
clf_RF_trigger = mf.prelim_RF(X_all_trigger,Y_all_clean,0.1)

##clf_stree_left = mf.prelim_single_tree(X_all_left,Y_all_clean,0.5)
##clf_stree_right = mf.prelim_single_tree(X_all_right,Y_all_clean,0.5)

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat3/')
count = 0
for file_name in os.listdir():
    count = count + 1
    print(count,file_name)
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

        TC_layer_left = mf.find_left(TC_layer,file)

        tt1 = 'TC' + str(TC_layer_left)
        tt2 = 'TC' + str(TC_layer_left + 20)

        LL1 = file.loc[:,tt1]
        LL2 = file.loc[:,tt2]

        TC_layer_right = mf.find_right(TC_layer,file)

        tt1 = 'TC' + str(TC_layer_right)
        tt2 = 'TC' + str(TC_layer_right + 20)

        LR1 = file.loc[:,tt1]
        LR2 = file.loc[:,tt2]

        TC_layer_opp = mf.find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]

        temp_x_trigger = mf.make_cont_x(L1,L2,ML,CP,CS,LO1,LO2,MW)
##        temp_x_left = mf.make_cont_x(LL1,LL2,ML,CP,CS,LO1,LO2,MW)
##        temp_x_right = mf.make_cont_x(LR1,LR2,ML,CP,CS,LO1,LO2,MW)
        
        logit_trigger = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger))
        logit_trigger_my = pd.DataFrame(my_logit_predict_proba(beta_0,beta,temp_x_trigger))
        RF_trigger = pd.DataFrame(clf_RF_trigger.predict_proba(temp_x_trigger))
        gbm_trigger = pd.DataFrame(clf_gbm_trigger.predict_proba(temp_x_trigger))
##        pred_left = pd.DataFrame(clf_stree_left.predict_proba(temp_x_left))
##        pred_right = pd.DataFrame(clf_stree_right.predict_proba(temp_x_right))

        plt.clf()
        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(3,1,1)
        plt.plot(range(len(logit_trigger.iloc[:,-1])),logit_trigger.iloc[:,-1],
                 range(len(logit_trigger.iloc[:,-1])),logit_trigger_my.iloc[:,-1],
                 range(len(gbm_trigger.iloc[:,-1])),gbm_trigger.iloc[:,-1],
                 range(len(gbm_trigger.iloc[:,-1])),RF_trigger.iloc[:,-1]
                 )
        plt.xlabel('Time')
        plt.ylabel('Probability of being a sticker')
        plt.legend(['logit','my_logit','GBM','RF'])
        plt.grid(1)

        plt.subplot(3,1,2)
        plt.plot(range(len(L2)),L2,range(len(L2)),LL2,range(len(L2)),LR2,range(len(L2)),L1)
        plt.legend(['Level 2','Left','Right','Level 1'])
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.grid(1)

        plt.subplot(3,1,3)
        plt.plot(range(len(ML)),ML)
        plt.xlabel('Time')
        plt.ylabel('Mold Level')
        plt.grid(1)

        plt.suptitle(file_name.split('.')[0] )
        plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//plot_cont//' + file_name.split('.')[0] + '.png'
        plt.savefig(plot_save)
        plt.close()
##
####
####
####

        
