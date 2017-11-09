import os
import numpy as np
import pandas as pd
import random
import pylab as plt
from tqdm import *
import BDS_module_17_10_17 as bds

##################################################################################################
##################################################################################################
##################################################################################################

def make_my_text(row):
    tt = ''
    col_names = list(row.keys())
    for i in range(len(row)):
        tt = tt + "\n"+ str(col_names[i]) + " : " +str(row[i])
    return tt


##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/first iteration/caster C old their alarms')

file = pd.read_excel("D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/caster_c_old_level_three_23_10_24_10.xlsx",sheetname='duplicates removed')
all_files = list(set(file.iloc[:,0]))
all_files.sort()
count = 0
n = 250

font = {'family': 'monospace',
        'size': 12,
        'weight' : 'light'
        }

for file_name in all_files:
    count+=1
    temp_file = file.loc[file.loc[:,'file name'] == file_name,:].reset_index().drop('index',1)
    read_name = file_name.split('_')[0]+'.csv'
    print(count,len(all_files),read_name)
    temp_read = pd.read_csv(read_name)
    mw = temp_read.loc[0,'M.width']
    for i in range(temp_file.shape[0]):
        print(i+1,temp_file.shape[0])
        self = temp_file.loc[i,'TC']
        left = bds.find_left(mw,self)
        right = bds.find_right(mw,self)
        
        tt1 = 'TC' + str(self)
        tt2 = 'TC' + str(self+20)
        tt3 = 'TC' + str(self+40)
        tt4 = 'TC' + str(self+60)
        if temp_file.loc[i,'time']-n < n:
            temp_time = range(0,temp_file.loc[i,'time']+50)
        else:
            temp_time = range(temp_file.loc[i,'time']-n,temp_file.loc[i,'time']+50)
        l1 = temp_read.loc[temp_time,tt1]
        l2 = temp_read.loc[temp_time,tt2]
        l3 = temp_read.loc[temp_time,tt3]
        l4 = temp_read.loc[temp_time,tt4]
        cs = temp_read.loc[temp_time,'C.speed']
        ml = temp_read.loc[temp_time,'M.level']
        
        fig = plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(232)
        plt.plot(temp_time,l1,temp_time,l2,temp_time,l3,temp_time,l4)
        plt.axvline(x=temp_file.loc[i,'time'],c='g',ls='--')
        if temp_file.loc[i,'time']-60 > 0:
            plt.axvline(x=temp_file.loc[i,'time']-60,c='g',ls='--')
        plt.grid(1,which='both')
        plt.minorticks_on()
        plt.title('trigger thermocouple')


        tt1 = 'TC' + str(left)
        tt2 = 'TC' + str(left+20)
        tt3 = 'TC' + str(left+40)
        tt4 = 'TC' + str(left+60)
        if temp_file.loc[i,'time']-n < n:
            temp_time = range(0,temp_file.loc[i,'time']+50)
        else:
            temp_time = range(temp_file.loc[i,'time']-n,temp_file.loc[i,'time']+50)
        l1 = temp_read.loc[temp_time,tt1]
        l2 = temp_read.loc[temp_time,tt2]
        l3 = temp_read.loc[temp_time,tt3]
        l4 = temp_read.loc[temp_time,tt4]
        cs = temp_read.loc[temp_time,'C.speed']
        ml = temp_read.loc[temp_time,'M.level']
        
        plt.subplot(231)
        plt.plot(temp_time,l1,temp_time,l2,temp_time,l3,temp_time,l4)
        plt.axvline(x=temp_file.loc[i,'time'],c='g',ls='--')
        if temp_file.loc[i,'time']-60 > 0:
            plt.axvline(x=temp_file.loc[i,'time']-60,c='g',ls='--')
        plt.grid(1,which='both')
        plt.minorticks_on()
        plt.title('left')

        tt1 = 'TC' + str(right)
        tt2 = 'TC' + str(right+20)
        tt3 = 'TC' + str(right+40)
        tt4 = 'TC' + str(right+60)
        if temp_file.loc[i,'time']-n < 0:
            temp_time = range(0,temp_file.loc[i,'time']+50)
        else:
            temp_time = range(temp_file.loc[i,'time']-n,temp_file.loc[i,'time']+50)
        l1 = temp_read.loc[temp_time,tt1]
        l2 = temp_read.loc[temp_time,tt2]
        l3 = temp_read.loc[temp_time,tt3]
        l4 = temp_read.loc[temp_time,tt4]
        cs = temp_read.loc[temp_time,'C.speed']
        ml = temp_read.loc[temp_time,'M.level']
        
        plt.subplot(234)
        plt.plot(temp_time,l1,temp_time,l2,temp_time,l3,temp_time,l4)
        plt.axvline(x=temp_file.loc[i,'time'],c='g',ls='--')
        if temp_file.loc[i,'time']-60 > 0:
            plt.axvline(x=temp_file.loc[i,'time']-60,c='g',ls='--')
        plt.grid(1,which='both')
        plt.minorticks_on()
        plt.title('right')

        plt.subplot(235)
        plt.plot(ml)
        plt.plot(100*cs)
        plt.axvline(x=temp_file.loc[i,'time'],c='g',ls='--')
        if temp_file.loc[i,'time']-60 > 0:
            plt.axvline(x=temp_file.loc[i,'time']-60,c='g',ls='--')
        plt.grid(1,which='both')
        plt.minorticks_on()

        ax = fig.add_subplot(233)
        my_text = make_my_text(temp_file.iloc[i,:])
        ax.minorticks_off()
        ax.axis('off')
        ax.text(0, 1, my_text,
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontdict=font)

        # super title is also taken as the name of the file
        sup_title = read_name.split('.')[0] + ' TC_' + str(self) + ' time_' + str(temp_file.loc[i,'time'])
        plt.suptitle(sup_title)
        plt.savefig('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/second iteration/caster C old 23 10/'+sup_title+'.png')
        plt.close()
