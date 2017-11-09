import os
import numpy as np
import pandas as pd
import pylab as plt

###################################################################################################
###################################################################################################
def deriv(y):
    tt = []
    for i in range(len(y)-1):
        tt.append(y[i+1]-y[i])
    return tt

###################################################################################################
###################################################################################################

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/first iteration/sticker data csv')
all_files = os.listdir()
tick_x = np.linspace(220,250,4)
temp=pd.DataFrame()
count = 0
for file_name in all_files:
    count = count + 1
    print(100*count/len(all_files))
    j = file_name.split('_')
    TC_layer = int(j[4]) + 1

    file = pd.read_csv(file_name)
    flag = 0
    
    tt = 'TC' + str(TC_layer + 20)
    y = file.loc[:,tt]
    x = range(300)

    plt.clf()
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(3,1,1)
    plt.plot(x,y)
    plt.title('Trigger ThermoCouple')
    plt.xticks([x for x in tick_x])
    plt.grid(1)
    
    y_d = deriv(y)
    x = range(299)

    plt.subplot(3,1,2)
    plt.plot(x,y_d)
    plt.title('Trigger Left')
    plt.xticks([x for x in tick_x])
    plt.grid(1)

    plt.subplot(3,1,3)
    x = range(300)
    plt.plot(x,file.loc[:,'M.level'],x,100*file.loc[:,'C.speed'])
    plt.grid(1)
    plt.title('Mould Level and Casting Speed')
    plt.xticks([x for x in tick_x])
    plt.legend(['Mould Level','100x Casting Speed'])
    
    sup_title = 'file name : ' + j[0] + ' ' + j[5] + ' Status : ' + j[6] + ' trigger TC : ' + str(TC_layer) + ' layer : ' + j[3] + ' Carbon % : ' + str(np.mean(file.loc[:,'C.percent'])) + ' Mold width : ' + str(np.mean(file.loc[:,'M.width']))
    plt.suptitle(sup_title)
    plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//plot_der//' + j[0] + '_'+ j[1]+ '_' + j[2]+ '_' + j[3]+ '_' + j[4]+ '_' + j[5]+ '_' + j[6] + '_'+ j[7] + '.png'
    plt.savefig(plot_save)
    plt.close()
