import os
import numpy as np
import pandas as pd
import random
import pylab as plt
from tqdm import *
import BDS_module_10_10_17 as bds

##################################################################################################
##################################################################################################
##################################################################################################

def start_index(temp_ref,xx):
    count_in = -1
    for i in xx:
        if i < temp_ref:
            count_in += 1
        else:
            break
    return count_in

def interpolated(ll,xx,n):
    tt = [ll[0]]
    temp_ref = [x+1 for x in range(1,n-1)]
    for i in temp_ref:
        x1 = xx[start_index(i,xx)]
        x2 = xx[start_index(i,xx)+1]
        y1 = ll[start_index(i,xx)]
        y2 = ll[start_index(i,xx)+1]

        qq = y1 + ((i-x1)*(y2-y1))/(x2-x1)
        tt.append(qq)
        
    tt.append(ll[-1])
    return tt

def my_scale(ll_1,ll_2):
    min_l = min(min(ll_1),min(ll_2))
    max_l = max(max(ll_1),max(ll_2))
    tt_1 = [10*(x-min_l)/(max_l-min_l) for x in ll_1]
    tt_2 = [10*(x-min_l)/(max_l-min_l) for x in ll_2]
    return tt_1,tt_2

def my_plot(l1,l2,ll1,ll2,li1,li2,xx,n):
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
##    plt.subplot(1,2,1)
##    plt.plot(l1)
##    plt.plot(l2)
##    plt.grid(1,which='both')
##    plt.minorticks_on()
##    plt.axvline(x=l2_index,c='g',ls='--')
##    plt.axvline(x=l1_index,c='r',ls='--')
##    plt.axvline(x=l1_index_rise,c='b',ls='--')
##    
##    plt.subplot(1,2,2)
    plt.plot(xx,ll1,'.-')
    plt.plot(xx,ll2,'.-')
    xx = np.linspace(1,n,n)
    plt.plot(xx,li1,'*-')
    plt.plot(xx,li2,'*-')
    
    plt.grid(1,which='both')
    plt.minorticks_on()
    plt.suptitle('file name : {0}, {1}, time : {2}, ratio : {3} / {4}'.format(read_name,tt1,l2_index,len(ll1),n))
    plt.savefig('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/second iteration/extracted_true/{0}_{1}_time_{2}.png'.format(read_name,tt1,l2_index))
    plt.close()
##    plt.show()
##
    pass

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/1_data')

file = pd.read_excel("D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/training_ones_level_three_05_10.xlsx",sheetname='raw data')
all_files = list(set(file.iloc[:,0]))
all_files.sort()
count = 0
tt = []
for file_name in all_files:
    count+=1
    temp_file = file.loc[file.loc[:,'file name'] == file_name,:].reset_index().drop('index',1)
##    read_name = file_name.split('_')[0]+'.csv'
    read_name = file_name+'.csv'
    print(count,len(all_files),read_name)
    temp_read = pd.read_csv(read_name)
    mw = temp_read.loc[0,'M.width']
    for i in range(temp_file.shape[0]):
##        print(i+1,temp_file.shape[0])
        self = temp_file.loc[i,'TC']
        
        tt1 = 'TC' + str(self)
        tt2 = 'TC' + str(self+20)
        l1 = list(temp_read.loc[:,tt1])
        l2 = list(temp_read.loc[:,tt2])
        l2_index = temp_file.loc[i,'time']
        l1_index_rise = bds.find_index_l1_rise(l1,l2_index)
        l1_index = bds.find_index_l1(l1,l2_index)
        
        n = 20

        ll1,ll2 = my_scale(l1[l1_index_rise-1:l2_index+1],l2[l1_index_rise-1:l2_index+1])
        
        xx = np.linspace(1,n,len(ll1))

        li1 = interpolated(ll1,xx,n)
        li2 = interpolated(ll2,xx,n)
##        my_plot(l1,l2,ll1,ll2,li1,li2,xx,n)
        li1.extend(li2)
        li1.append(1)

        tt.append(li1)

tt = pd.DataFrame(tt)
        
tt.to_csv('for_ANN_1.csv',index=False)
