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

###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_3_3_17.csv')
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

X_all_left = pd.DataFrame(pp.scale(temp.iloc[:,0:17]))
X_all_trigger = pd.DataFrame(pp.scale(temp.iloc[:,17:34]))
X_all_right = pd.DataFrame(pp.scale(temp.iloc[:,34:51]))
Y_all_clean = temp.iloc[:,51]
index_file_clean = temp.iloc[:,52]

X_all_trigger.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17"]
X_all_left.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17"]
X_all_right.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17"]

print(X_all_trigger.shape,' : Size of all X trigger')
print(X_all_left.shape,' : Size of all X left')
print(X_all_right.shape,' : Size of all X right')
print(Y_all_clean.shape,' : Size of all Y')
print(index_file_clean.shape,' : Shape of index file')

temp = X_all_trigger.apply(include_tree,axis=1)

data_logit = pd.concat([X_all_trigger,temp],axis=1)

print(X_all_trigger.shape)
print(data_logit.shape)

global woe_table
woe_table = mf.calc_WOE(data_logit.iloc[:,-1],Y_all_clean)

temp2 = data_logit.apply(subs_WOE,axis=1)

# just to make sure everything is going okay
temp3 = pd.concat([temp2,data_logit.iloc[:,-1]],axis=1)

data_logit = pd.concat([X_all_trigger,temp2],axis=1)

##clf_logit = mf.prelim_logit(data_logit,Y_all_clean,0.1,1)

##clf_gbm = mf.prelim_gbm(X_all_trigger,Y_all_clean,0.1)
