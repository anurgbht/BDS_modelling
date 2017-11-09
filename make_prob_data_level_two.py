import os
import numpy as np
import pandas as pd
import BDS_module_17_10_17 as bds
import random
import pylab as plt
from tqdm import tqdm

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test')
all_files = os.listdir()
##all_files = all_files[:1]
count = 0

for file_name in all_files:
    count += 1
##    file_name = 'gb290245.csv'
    print(file_name,count,len(all_files))
##    print('\n')
    file = pd.read_csv(file_name)
    file = file.reset_index().drop('index',1)
    mw = file.loc[0,'M.width']
    active_list = bds.find_active(mw)
    ml  = file.loc[:,'M.level']
    cs = file.loc[:,'C.speed']
    prob_temp12 = []
    prob_temp23 = []
    for i in range(10,file.shape[0]):
        if i<60:
            temp = file.iloc[:i,:]
        else:
            temp = file.iloc[i-60:i,:]
        prob_temp12.append(bds.prob_level_two(temp,active_list,'12'))
        prob_temp23.append(bds.prob_level_two(temp,active_list,'23'))
    prob_temp12 = pd.DataFrame(prob_temp12)
    prob_temp23 = pd.DataFrame(prob_temp23)
    
##    prob_temp.to_csv('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/input_for_level_two_zeros/'+file_name.split('.')[0]+'_level_two.csv',index=False)
##
    x = range(prob_temp12.shape[0])
##    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    for i in range(prob_temp12.shape[1]):
        p12 = list(prob_temp12.iloc[:,i])
        p23 = list(prob_temp23.iloc[:,i])
        plt.subplot(6,4,i+1)
        plt.plot(x,p12,x,p23)
        plt.ylim([0,1])
        plt.yticks([0.1,0.4,0.5,0.9])
        plt.grid(1)
        plt.ylabel(active_list[i])

    x = range(len(ml))
    plt.subplot(6,4,i+2)
    plt.plot(x,ml,x,100*cs)
    plt.yticks([20,60,100])
    plt.grid(1)
    plt.suptitle(file_name)

##    plt.savefig('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/second iteration/caster B recent pred 29 9/'+file_name.split('.')[0]+'.png')
##    plt.close()
    plt.show()

    x = range(file.shape[0])
##    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    for i in range(len(active_list)):
        l1 = file.loc[:,'TC'+str(active_list[i])]
        l2 = file.loc[:,'TC'+str(active_list[i]+20)]
        l3 = file.loc[:,'TC'+str(active_list[i]+40)]
        plt.subplot(6,4,i+1)
        plt.plot(x,l1,x,l2,x,l3)
        plt.ylabel(active_list[i])
        plt.grid(1)

    x = range(len(ml))
    plt.subplot(6,4,i+2)
    plt.plot(x,ml,x,100*cs)
    plt.grid(1)
##    plt.suptitle(file_name)

##    plt.savefig('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/second iteration/kalinganagar breakout/'+file_name.split('.')[0]+'_temp.png')
##    plt.close()
    plt.show()

