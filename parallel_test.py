import multiprocessing as mp
import time
import numpy as np
import random
import os
import pandas as pd
import pickle

def include_tree(row):
    x3 = row[2]
    x10 = row[9]
    x16 = row[15]
    x14 = row[13]
    x17 = row[16]
    x3 = row[2]

    if x3 <= 8.345:
        if x3 <= 5.25:
            ## A
            tt = -4.627443
        elif x3 > 5.25:
            if x14 <= -1.445:
                if x17 <= 0.045:
                    ## B
                    tt = 2.141884
                elif x17 > 0.045:
                    ## C
                    tt = 0.247894
            elif x14 > -1.445:
                ## D
                tt = -1.716035
    elif x3 > 8.345:
        if x16 <= 9.45:
            if x14 <= -0.405:
                ## E
                tt = 1.991644
            elif x14 > -0.405:
                ## F
                tt = 0.043099
        elif x16 > 9.45:
            ## G
            tt = 4.45875
        
    return tt
            
def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp

def feature_calc(x,ml,CP,cs,logistic_model):
    import my_functions_no_keras as mf
    l1 = x[0]
    l2 = x[1]
    l3 = x[2]
    l4 = x[3]
    print(len(l1), " file size")
    for i in range(len(l1)-63):
        L1 = l1.iloc[i:i+61]
        L2 = l2.iloc[i:i+61]
        L3 = l3.iloc[i:i+61]
        L4 = l4.iloc[i:i+61]
        ML = ml.iloc[i:i+61]
        CS = cs.iloc[i:i+61]

        temp_x1 = mf.make_one_x(L1,L2,ML,CP,CS)
        # to include weight of evidence calculations here
        temp_x1[20] = include_tree(list(temp_x1))

        temp_x2 = mf.make_one_x(L2,L3,ML,CP,CS)
        # to include weight of evidence calculations here
        temp_x2[20] = include_tree(list(temp_x2))

        temp_x3 = mf.make_one_x(L3,L4,ML,CP,CS)
        # to include weight of evidence calculations here
        temp_x3[20] = include_tree(list(temp_x3))

        prob1 = logistic_model.predict_proba(temp_x1)
        prob2 = logistic_model.predict_proba(temp_x2)
        prob3 = logistic_model.predict_proba(temp_x3)
    
    return prob1[0][1], prob2[0][1], prob3[0][1]

def make_list(window_file,active_list):
    temp = []
    for TC in active_list:
        l1 = 'TC' + str(TC)
        l2 = 'TC' + str(TC+20)
        l3 = 'TC' + str(TC+40)
        l4 = 'TC' + str(TC+60)
        temp.append([window_file.loc[:,l1],window_file.loc[:,l2],window_file.loc[:,l3],window_file.loc[:,l4]])
    return temp

def apply_async_TC(window_file,logistic_model):
    import my_functions_no_keras as mf
    ML = window_file.loc[:,'M.level']
    CS = window_file.loc[:,'C.speed']
    CP = window_file.loc[0,'C.percent']
    MW = window_file.loc[0,'M.width']
    active_list = mf.find_active(MW)
    
    TC_dat = make_list(window_file,active_list)
    
    with mp.Pool(processes=20) as pool :
        print("entering the parallel code")
        results = [pool.apply_async(feature_calc, args=(x,ML,CP,CS,logistic_model)) for x in TC_dat]
        output = [p.get() for p in results]
        print(output)

def apply_for_TC(window_file,logistic_model):
    import my_functions_no_keras as mf
    ML = window_file.loc[:,'M.level']
    CS = window_file.loc[:,'C.speed']
    CP = window_file.loc[0,'C.percent']
    MW = window_file.loc[0,'M.width']
    active_list = mf.find_active(MW)
    
    TC_dat = make_list(window_file,active_list)
    
    print("entering the sequential code")
    results = [feature_calc(x,ML,CP,CS,logistic_model) for x in TC_dat]
    print(results)

###############################################################################
###############################################################################
###############################################################################

if __name__ == '__main__':
    import my_functions_no_keras as mf
    os.chdir("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/test/")
    file_list = os.listdir()
    with open("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/logistic_pickle.pickle",'rb') as f:
        logistic_model = pickle.load(f)

    for file_name in file_list:
        print(file_name)
        file = pd.read_csv(file_name)
##        file = file.iloc[0:260,:]

##        start_time = time.time()
##        apply_for_TC(file,logistic_model)
##        end_time = time.time()
##        print(end_time-start_time)
        
        start_time = time.time()
        apply_async_TC(file,logistic_model)
        end_time = time.time()
        print(end_time-start_time)


