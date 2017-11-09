import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BDS_module_17_10_17 as bds

################################################################################################
################################################################################################
################################################################################################

orginal_path = os.getcwd()
os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/alarm files/Caster3 Apr-Sep15 csv')
all_files = os.listdir()
tick_x = np.linspace(220,250,4)
temp=pd.DataFrame()
count = 0
for file_name in all_files:
    count = count + 1
    print(count)
    TC_layer = int(file_name.split('_')[4].split('.')[0]) + 1
    
    file = pd.read_csv(file_name)
    tt1 = 'TC' + str(TC_layer)
    tt1 = 'TC' + str(TC_layer)
    tt2 = 'TC' + str(TC_layer + 20)
    tt3 = 'TC' + str(TC_layer + 40)
    tt4 = 'TC' + str(TC_layer + 60)
    
    y1 = file.loc[:,tt1]
    y2 = file.loc[:,tt2]
    y3 = file.loc[:,tt3]
    y4 = file.loc[:,tt4]
    x = range(len(y1))
    plt.clf()
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(4,1,1)
    plt.plot(x,y1,x,y2,x,y3,x,y4)
    plt.title('Trigger ThermoCouple')
    plt.xticks([x for x in tick_x])
    plt.legend(['Level 1','Level 2','Level 3','Level 4'])
    plt.grid(1)

    TC_layer_left = bds.find_left(file.loc[0,'M.width'],TC_layer)
    
    tt1 = 'TC' + str(TC_layer_left)
    tt2 = 'TC' + str(TC_layer_left + 20)
    tt3 = 'TC' + str(TC_layer_left + 40)
    tt4 = 'TC' + str(TC_layer_left + 60)
    layer = [tt1,tt2,tt3,tt4]
    
    y1 = file.loc[:,tt1]
    y2 = file.loc[:,tt2]
    y3 = file.loc[:,tt3]
    y4 = file.loc[:,tt4]
    x = np.linspace(0,300,len(y1))

    plt.subplot(4,1,2)
    plt.plot(x,y1,x,y2,x,y3,x,y4)
    plt.title('Trigger Left')
    plt.xticks([x for x in tick_x])
    plt.legend(['Level 1','Level 2','Level 3','Level 4'])
    plt.grid(1)

    TC_layer_right = bds.find_right(file.loc[0,'M.width'],TC_layer)
    
    tt1 = 'TC' + str(TC_layer_right)
    tt2 = 'TC' + str(TC_layer_right + 20)
    tt3 = 'TC' + str(TC_layer_right + 40)
    tt4 = 'TC' + str(TC_layer_right + 60)
    layer = [tt1,tt2,tt3,tt4]
    
    y1 = file.loc[:,tt1]
    y2 = file.loc[:,tt2]
    y3 = file.loc[:,tt3]
    y4 = file.loc[:,tt4]
    x = np.linspace(0,300,len(y1))

    plt.subplot(4,1,3)
    plt.plot(x,y1,x,y2,x,y3,x,y4)
    plt.title('Trigger Right')
    plt.xticks([x for x in tick_x])
    plt.legend(['Level 1','Level 2','Level 3','Level 4'])
    plt.grid(1)

    plt.subplot(4,1,4)
    plt.plot(x,file.loc[:,'M.level'],x,100*file.loc[:,'C.speed'])
    plt.grid(1)
    plt.title('Mould Level and Casting Speed')
    plt.xticks([x for x in tick_x])
    plt.legend(['Mould Level','100x Casting Speed'])

    sup_title = 'file name : ' + file_name +  ' Carbon % : ' + str(np.mean(file.loc[:,'C.percent'])) + ' Mold width : ' + str(np.mean(file.loc[:,'M.width']))

    plt.suptitle(sup_title)

    plot_save = 'D://OneDrive - Tata Insights and Quants, A division of Tata Industries//Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//second iteration//caster C old plots//' + file_name.split('.')[0] + '.png'
    plt.savefig(plot_save)
    plt.close()
