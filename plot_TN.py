import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import implement_functions as mf
import numpy as np

def plot_fun_my(file,file_pred):
    new_folder =  '/mnt2/client16/TataSteel_BDS_ProcDat/prelim_analysis/plots/sample_plots/' + file_name.split('.')[0]
    os.mkdir(new_folder)
    for i in range(20):
        TC1 = i+1
        TC2 = i+21
        TC3 = i+41
        TC4 = i+61
        L1 = file.loc[:,TC1]
        L2 = file.loc[:,TC2]
        L3 = file.loc[:,TC3]
        L4 = file.loc[:,TC4]

        p1 = list(np.zeros(63))
        p2 = list(np.zeros(63))

        p1.extend(list(file_pred.iloc[:,2*i+0]))
        p2.extend(list(file_pred.iloc[:,2*i+1]))

        temp_x = range(len(L1))
        temp_x1 = range(len(p1))

        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(3,1,1)
        plt.plot(temp_x,L1,temp_x,L2,temp_x,L3,temp_x,L4)
        plt.legend(['Layer 1','Layer 2','Layer 3','Layer 4'])
        plt.grid(1)

        plt.subplot(3,1,2)
        plt.plot(temp_x1,p1,temp_x1,p2)
        plt.legend(['L1-L2','L2-L3'])
        plt.yticks([0.5,0.6,0.7,0.8,0.9,0.95,1])
        plt.grid(1)

        plt.subplot(3,1,3)
        plt.plot(temp_x,file.iloc[:,-3],temp_x,100*file.iloc[:,-4])
        plt.legend(['Mold level','100x Casting Speed'])
        plt.grid(1)

        
        plt.suptitle(file_name.split('.')[0]+'_TC_'+str(i+1))
        plot_save = new_folder + '//'+ file_name.split('.')[0] + '_'+ str(i+1)+ '.png'
        plt.savefig(plot_save)
        plt.close()
##        plt.show()
################################################################################################
################################################################################################
################################################################################################
predicted_path = '/mnt2/client16/TataSteel_BDS_ProcDat/prelim_analysis/data/validation/splits/'
os.chdir('/mnt2/bkumar/steel/input_backup')
file_list = os.listdir()
count_iter = 0
##random.shuffle(file_list)
for file_name in file_list:
    count_iter += 1
    if file_name != 'input.txt':
        print(file_name,count_iter,len(file_list))
        file = pd.read_csv(file_name,header=None)
        file_pred = pd.read_csv(predicted_path+file_name.split('.')[0]+'_predicted.csv',sep='|',header=None)
        print(file.shape)
    ##    file = file.iloc[1400:1700,:]
        plot_fun_my(file,file_pred)
