import os
import numpy as np
import pandas as pd
import implement_functions_23_08_17 as mf
import random
import pylab as plt
from tqdm import tqdm
import time

##################################################################################################
##################################################################################################
##################################################################################################

def my_filter(row):

    if ((row['self']>0.6) and (row['ratio'] > 1.5) and (row['cross'] > 0.4)):
        a1 = 1
    else:
        a1 = 0
        
    if ((row['self']>0.6) and (row['ratio'] > 1.5) and (row['opp'] > 0.1)):
        a2 = 1
    else:
        a2 = 0
        
    if ((row['self']>0.6) and (row['opp'] > 0.1) and (row['cross'] > 0.4)):
        a3 = 1
    else:
        a3 = 0

    if (((a1>0)and row['cs change'] > 0.1) or (row['cs change'] <= 0.1)):
        a4 = 1
    else:
        a4 = 0
    a_all = a1 or a2 or a3
    return pd.Series({'a1':a1,'a2':a2,'a3':a3,'a_all':a_all,'a4':a4})

def remove_consec(dat):
    print(len(set(dat.loc[:,'file name'])))
    for file_name in set(dat.loc[:,'file name']):
        dat_temp = dat.loc[dat.loc[:,'file name'] == file_name]
        if dat_temp.shape[0]>1:
            pass
        else:
            
##        print(dat_temp.shape)

        
##################################################################################################
##################################################################################################
##################################################################################################

file = pd.read_csv('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/caster_b_recent_level_three_18_09.csv')

file.columns = ["file name","time","TC","self2","std1","std2","l1 drop","l2 drop","cross2","l1 avg","cs change","ml change","high freq","neighbor","self","opp","rise slope l1","rise slope l2","cross","rise amp l1","rise amp l2","drop slope l1","drop amp l1","rise time","drop time","time delta"]
##file  = file.iloc[:2,:]
file.loc[:,'ratio'] = file.loc[:,'std1']/file.loc[:,'std2']

tt = file.apply(my_filter,axis=1)

temp = pd.concat([file,tt],axis=1)

filter1 = temp.loc[temp.loc[:,'a_all']==1,:]
filter1 = filter1.reset_index().drop('index',1)
print(filter1.shape)

filter2 = filter1.loc[filter1.loc[:,'a4']==1]
filter2 = filter2.reset_index().drop('index',1)
print(filter2.shape)

filter3 = remove_consec(filter2)
