import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import implement_functions as mf

def plot_fun_my(file):
    new_folder =  'D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plots/caster_b_plots/' + file_name.split('.')[0]
    os.mkdir(new_folder)
    for i in range(20):
        TC1 = 'TC'+str(i+1)
        TC2 = 'TC'+str(i+21)
        TC3 = 'TC'+str(i+41)
        TC4 = 'TC'+str(i+61)
        L1 = file.loc[:,TC1]
        L2 = file.loc[:,TC2]
        L3 = file.loc[:,TC3]
        L4 = file.loc[:,TC4]

        temp_x = range(len(L1))

        plt.figure(num=None, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(2,1,1)
        plt.plot(temp_x,L1,temp_x,L2,temp_x,L3,temp_x,L4)
        plt.grid(1)

        plt.subplot(2,1,2)
        plt.plot(temp_x,file.loc[:,'M.level'],temp_x,100*file.loc[:,'C.speed'])
        plt.grid(1)

        
        plt.suptitle(file_name.split('.')[0]+'_TC_'+str(i+1))
        plot_save = new_folder + '//'+ file_name.split('.')[0] + '_'+ str(i+1)+ '.png'
        plt.savefig(plot_save)
        plt.close()
##        plt.show()
################################################################################################
################################################################################################
################################################################################################
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/caster B recent')
file_list = os.listdir()

tt = pickle.load(open('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/cs_drop_files.pkl','rb'))
count_iter = 0
for file_name in file_list:
    count_iter += 1
    print(100*count_iter/len(file_list))
    if file_name in tt:
        file = pd.read_csv(file_name)
        plot_fun_my(file)
