import os
import numpy as np
import pandas as pd
import implement_functions as mf
import random
import pylab as plt

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/Alarm Filecc2 Oct- Feb17 csv')
all_files = os.listdir()
count_miss = 0
count = 0
for file_name in all_files:
    count += 1
    print(count,len(all_files))
    file = pd.read_csv(file_name)
    mw = file.loc[0,'M.width']
    # numbering 0-19 in file
    self = int(file_name.split('_')[4].split('.')[0])+1
    tt1 = 'TC' + str(self)
    tt2 = 'TC' + str(self+20)
    tt3 = 'TC' + str(self+40)
    tt4 = 'TC' + str(self+60)
    
    l1 = file.loc[:,tt1]
    l2 = file.loc[:,tt2]
    l3 = file.loc[:,tt3]
    l4 = file.loc[:,tt4]
    ml  = file.loc[:,'M.level']
    cs = file.loc[:,'C.speed']
    x = range(len(l1))

    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(4,1,2)
    plt.plot(x,l1,x,l2,x,l3,x,l4)
    plt.ylabel('Self')
    plt.grid(1)

    left = mf.find_left(mw,self)
    tt1 = 'TC' + str(left)
    tt2 = 'TC' + str(left+20)
    tt3 = 'TC' + str(left+40)
    tt4 = 'TC' + str(left+60)
    l1 = file.loc[:,tt1]
    l2 = file.loc[:,tt2]
    l3 = file.loc[:,tt3]
    l4 = file.loc[:,tt4]
    
    plt.subplot(4,1,1)
    plt.plot(x,l1,x,l2,x,l3,x,l4)
    plt.ylabel('Left')
    plt.grid(1)
    
    right = mf.find_right(mw,self)
    tt1 = 'TC' + str(right)
    tt2 = 'TC' + str(right+20)
    tt3 = 'TC' + str(right+40)
    tt4 = 'TC' + str(right+60)
    l1 = file.loc[:,tt1]
    l2 = file.loc[:,tt2]
    l3 = file.loc[:,tt3]
    l4 = file.loc[:,tt4]
    
    plt.subplot(4,1,3)
    plt.plot(x,l1,x,l2,x,l3,x,l4)
    plt.ylabel('Right')
    plt.grid(1)

    plt.subplot(4,1,4)
    plt.plot(x,ml,x,100*cs)
    plt.grid(1)

    plt.suptitle(file_name)
    plt.savefig('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/caster B recent alarms/'+file_name.split('.')[0]+'.png')
    plt.close()
