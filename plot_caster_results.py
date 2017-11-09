import os
import numpy as np
import pandas as pd
import implement_functions as mf
import random
import pylab as plt

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/first iteration/caster C recent')

plot_file = pd.read_csv('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/results/caster_C_plot.csv')
count_miss = 0
for i in range(plot_file.shape[0]):
    try:
        print(i+1,plot_file.iloc[i,0])
        file = pd.read_csv(plot_file.iloc[i,0]+'.csv')
        file.columns = ["TimesStamp","TC1","TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10","TC11","TC12","TC13","TC14","TC15","TC16","TC17","TC18","TC19","TC20","TC21","TC22","TC23","TC24","TC25","TC26","TC27","TC28","TC29","TC30","TC31","TC32","TC33","TC34","TC35","TC36","TC37","TC38","TC39","TC40","TC41","TC42","TC43","TC44","TC45","TC46","TC47","TC48","TC49","TC50","TC51","TC52","TC53","TC54","TC55","TC56","TC57","TC58","TC59","TC60","TC61","TC62","TC63","TC64","TC65","TC66","TC67","TC68","TC69","TC70","TC71","TC72","TC73","TC74","TC75","TC76","TC77","TC78","TC79","TC80","C.speed","M.level","M.width","C.percent"]
        mw = file.loc[0,'M.width']
        self = plot_file.iloc[i,1]
        tt1 = 'TC' + str(self)
        tt2 = 'TC' + str(self+20)
        tt3 = 'TC' + str(self+40)
        tt4 = 'TC' + str(self+60)
        time_index = range(plot_file.iloc[i,2] - 350,plot_file.iloc[i,2] + 350)

        l1 = file.loc[time_index,tt1]
        l2 = file.loc[time_index,tt2]
        l3 = file.loc[time_index,tt3]
        l4 = file.loc[time_index,tt4]
        ml  = file.loc[time_index,'M.level']
        cs = file.loc[time_index,'C.speed']
        x = range(len(l1))

        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(4,1,1)
        plt.plot(x,l1,x,l2,x,l3,x,l4)
        plt.ylabel('Self')
        plt.grid(1)

        left = mf.find_left(mw,self)
        tt1 = 'TC' + str(left)
        tt2 = 'TC' + str(left+20)
        tt3 = 'TC' + str(left+40)
        tt4 = 'TC' + str(left+60)
        l1 = file.loc[time_index,tt1]
        l2 = file.loc[time_index,tt2]
        l3 = file.loc[time_index,tt3]
        l4 = file.loc[time_index,tt4]
        
        plt.subplot(4,1,2)
        plt.plot(x,l1,x,l2,x,l3,x,l4)
        plt.ylabel('Left')
        plt.grid(1)
        
        right = mf.find_right(mw,self)
        tt1 = 'TC' + str(right)
        tt2 = 'TC' + str(right+20)
        tt3 = 'TC' + str(right+40)
        tt4 = 'TC' + str(right+60)
        l1 = file.loc[time_index,tt1]
        l2 = file.loc[time_index,tt2]
        l3 = file.loc[time_index,tt3]
        l4 = file.loc[time_index,tt4]
        
        plt.subplot(4,1,3)
        plt.plot(x,l1,x,l2,x,l3,x,l4)
        plt.ylabel('Right')
        plt.grid(1)

        plt.subplot(4,1,4)
        plt.plot(x,ml,x,100*cs)
        plt.grid(1)

        plt.suptitle(plot_file.iloc[i,0])
        plt.savefig('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/caster C alarms raised/'+plot_file.iloc[i,0]+'.png')
        plt.close()
    except:
        count_miss += 1
        print('Missed a file',count_miss)
