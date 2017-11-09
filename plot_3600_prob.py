import os
import numpy as np
import pandas as pd
import implement_functions_new_X as mf
import random
import pylab as plt
from tqdm import tqdm

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/0_data')
all_files = os.listdir()
##all_files = ["F1010708_094_01_2_09_CASTER-B_True_JAN'15.csv"]
##count = 0

for file_name in tqdm(all_files):
##    count += 1
##    print(count,len(all_files))
    file = pd.read_csv(file_name)
    mw = file.loc[0,'M.width']
    active_list = mf.find_active(mw)
    x = range(file.shape[0])
    flag = 0
    ml  = file.loc[:,'M.level']
    cs = file.loc[:,'C.speed']
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    for i in range(len(active_list)):
        self = active_list[i]
        tt1 = 'TC' + str(self)
        tt2 = 'TC' + str(self+20)
        tt3 = 'TC' + str(self+40)
        
        l1 = file.loc[:,tt1]
        l2 = file.loc[:,tt2]
        l3 = file.loc[:,tt3]
        p12 = mf.make_parallel_prob(l1,l2,ml,cs)
        p23 = mf.make_parallel_prob(l2,l3,ml,cs)
        if ((max(p12)>0.5)or(max(p23)>0.5)):
            flag = 1

        plt.subplot(6,4,i+1)
        plt.plot(x,p12,x,p23)
        plt.ylim([0,1])
        plt.grid(1)

    plt.subplot(6,4,i+2)
    plt.plot(x,ml,x,100*cs)
    plt.yticks([20,100])
    plt.grid(1)

    plt.suptitle(file_name)
    if flag == 1:
        plt.savefig('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/caster B alarms oct-feb17 true false/online test prob/'+file_name.split('.')[0]+'_prob.png')
    plt.close()

    if flag == 1:
        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
        for i in range(len(active_list)):
            self = active_list[i]
            tt1 = 'TC' + str(self)
            tt2 = 'TC' + str(self+20)
            tt3 = 'TC' + str(self+40)
            
            l1 = file.loc[:,tt1]
            l2 = file.loc[:,tt2]
            l3 = file.loc[:,tt3]
            
            plt.subplot(6,4,i+1)
            plt.plot(x,l1,x,l2,x,l3)
            plt.grid(1)

        plt.subplot(6,4,i+2)
        plt.plot(x,ml,x,100*cs)
        plt.yticks([20,100])
        plt.grid(1)

        plt.suptitle(file_name)
        plt.savefig('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/caster B alarms oct-feb17 true false/online test prob/'+file_name.split('.')[0]+'.png')
        plt.close()

