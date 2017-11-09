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
    x16 = row[15]

    if (x2<=0.1793):
        if (x7<=-0.7124):
            if (x6<=0.1891):
                tt = 'A'
            elif(x6>0.1891):
                tt = 'B'
        elif(x7>-0.7124):
            if(x6<=-0.3407):
                tt = 'D'
            elif(x6>-0.3407):
                tt = 'D'
    elif(x2>0.1793):
        if(x6<=-0.5404):
            tt = 'E'
        elif(x6>-0.5404):
            if(x7<=-0.256):
                if(x16<=0.1502):
                    tt = 'F'
                elif(x16>0.1502):
                    tt = 'G'
            elif(x7>-0.256):
                tt = 'H'
    return tt

###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_28_2_17.csv')
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

X_all_left = pd.DataFrame(pp.scale(temp.iloc[:,0:16]))
X_all_trigger = pd.DataFrame(pp.scale(temp.iloc[:,16:32]))
X_all_right = pd.DataFrame(pp.scale(temp.iloc[:,32:48]))
Y_all_clean = temp.iloc[:,48]
index_file_clean = temp.iloc[:,49]

X_all_trigger.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16"]
X_all_left.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16"]
X_all_right.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16"]

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
woe_table = calc_WOE(data_logit.iloc[:,-1],Y_all_clean)

temp2 = data_logit.apply(subs_WOE,axis=1)

# just to make sure everything is going okay
temp3 = pd.concat([temp2,data_logit.iloc[:,-1]],axis=1)

data_logit = pd.concat([X_all_trigger,temp2],axis=1)

clf_logit = prelim_logit(data_logit,Y_all_clean)
