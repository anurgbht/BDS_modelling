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
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat2')
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
    print(count,"/",ttt)
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
        tag = file_name.split('_')[6]
        for i in range(4):
            l1 = L1.iloc[i:61*i + 61]
            l2 = L2.iloc[i:61*i + 61]
            temp1 = np.std(l1.iloc[-15:].reset_index().iloc[:,-1])
            temp2 = np.std(l2.iloc[-15:].reset_index().iloc[:,-1])
            if (i == 3) and (tag == 'True'):
                tt = [temp1,temp2,1]
                temp = temp.append([tt])
            else:
                tt = [temp1,temp2,0]
                temp = temp.append([tt])
            
plt.scatter(temp.iloc[:,0],temp.iloc[:,1],c=temp.iloc[:,2])
plt.grid()
plt.show()

