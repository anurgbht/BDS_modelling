import os
import pandas as pd
import numpy as np
import my_functions as mf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
import time
import pickle


def include_tree(row):
    x3 = row[2]
    x10 = row[9]
    x16 = row[15]
    x14 = row[13]
    x17 = row[16]
    x3 = row[2]
    x3 = row[2]

    if x3 <= 8.345:
        if x3 <= 5.25:
            tt = 'A'
        elif x3 > 5.25:
            if x14 <= -1.445:
                if x17 <= 0.045:
                    tt = 'B'
                elif x17 > 0.045:
                    tt = 'C'
            elif x14 > -1.445:
                tt = 'D'
    elif x3 > 8.345:
        if x16 <= 9.45:
            if x14 <= -0.405:
                tt = 'E'
            elif x14 > -0.405:
                tt = 'F'
        elif x16 > 9.45:
            tt = 'G'
        
    return tt
            

def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp


###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')
with open('logistic_pickle.pickle','rb') as f:
    clf_logit_trigger = pickle.load(f)



###########################################################################################
###################################Continuous Analysis######################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/BDS Data/online files/')
##file_list = os.listdir()
file_list=['gc312349.csv']

for file_name in file_list:
    print(file_name)
    file = pd.read_csv(file_name)
    ML = file.loc[:,'M.level']
    CS = file.loc[:,'C.speed']
    CP = file.loc[:,'C.percent']
    MW = file.loc[:,'M.width']
    active_range = mf.find_active(MW[0])
    count = 0
    for TC_layer in active_range:
        count += 1
        print(count,len(active_range))
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]

        TC_layer_opp = mf.find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]

        # making the continuous x's
        temp_x_trigger = mf.make_cont_x(L1,L2,ML,CP,CS,LO1,LO2,MW)
        # including the tree based information
        temp = temp_x_trigger.apply(include_tree,axis=1)
        temp_data = pd.concat([temp_x_trigger,temp],axis=1)
        temp2 = temp_data.apply(subs_WOE,axis=1)
        temp_x_trigger = pd.concat([temp_x_trigger,temp2],axis=1)
        
        logit_trigger = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger))
