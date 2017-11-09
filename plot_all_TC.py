import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/first iteration/sticker data csv')
all_files = os.listdir()
random.shuffle(all_files)
tick_x = [240]
temp=pd.DataFrame()
count = 0
temp_x = range(300)

for file_name in all_files:
    if (file_name != 'file_modifications.py') & (file_name != 'new_dat') & (file_name != 'plots'):
        count = count + 1
        print(count)

        if count < 200 :

            file = pd.read_csv(file_name)

            if np.average(file.layer) == 2:
                temp = file.loc[:,['TC21', 'TC22', 'TC23', 'TC24','TC25', 'TC26', 'TC27', 'TC28', 'TC29', 'TC30', 'TC31', 'TC32', 'TC33','TC34', 'TC35', 'TC36', 'TC37', 'TC38', 'TC39', 'TC40']]

                plt.clf()
                plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')

                plt.subplot(2,3,1)
                plt.plot(temp_x,temp.iloc[:,0],temp_x,temp.iloc[:,1],temp_x,temp.iloc[:,2],temp_x,temp.iloc[:,3],temp_x,temp.iloc[:,4])
                plt.ylabel('Temperature')
                plt.xticks([x for x in tick_x])
                plt.legend(['TC1','TC2','TC3','TC4','TC5'])
                plt.grid(1)

                plt.subplot(2,3,2)
                plt.plot(temp_x,temp.iloc[:,5],temp_x,temp.iloc[:,6],temp_x,temp.iloc[:,7],temp_x,temp.iloc[:,8],temp_x,temp.iloc[:,9])
                plt.ylabel('Temperature')
                plt.xticks([x for x in tick_x])
                plt.legend(['TC6','TC7','TC8','TC9','TC10'])
                plt.grid(1)

                plt.subplot(2,3,3)
                plt.plot(temp_x,temp.iloc[:,10],temp_x,temp.iloc[:,11],temp_x,temp.iloc[:,12],temp_x,temp.iloc[:,13],temp_x,temp.iloc[:,14])
                plt.ylabel('Temperature')
                plt.xticks([x for x in tick_x])
                plt.legend(['TC11','TC12','TC13','TC14','TC15'])
                plt.grid(1)

                plt.subplot(2,3,4)
                plt.plot(temp_x,temp.iloc[:,15],temp_x,temp.iloc[:,16],temp_x,temp.iloc[:,17],temp_x,temp.iloc[:,18],temp_x,temp.iloc[:,19])
                plt.ylabel('Temperature')
                plt.xticks([x for x in tick_x])
                plt.legend(['TC16','TC17','TC18','TC19','TC20'])
                plt.grid(1)

                TC_no = 'TC'+str(file.TC_no[0])

                plt.subplot(2,3,5)
                plt.plot(temp_x,file.loc[:,TC_no],temp_x, (np.sum(temp,axis=1)-file.loc[:,TC_no])/19,temp_x, file.loc[:,TC_no] - (np.sum(temp,axis=1)-file.loc[:,TC_no])/19)
                plt.ylabel('Temperature')
                plt.xticks([x for x in tick_x])
                plt.legend(['trigger','mean','difference'])
                plt.grid(1)

                plt.subplot(2,3,6)
                plt.plot(temp_x,file.loc[:,'M.level'],temp_x,100*file.loc[:,'C.speed'])
                plt.xticks([x for x in tick_x])
                plt.legend(['Mold level','100x Casting Speed'])
                plt.grid(1)

                plt.suptitle(file_name)

                plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//plot_temp//' + file_name.split('.')[0] + '.png'
                plt.savefig(plot_save)
                plt.close()


