import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
import random

def find_left(TC_layer,file):
    flag = 0
    file = file.loc[0:240,["TC1","TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10","TC11","TC12","TC13","TC14","TC15","TC16","TC17","TC18","TC19","TC20"]]
    tt1 = file.mean(axis=0)
    # there is already a plus one, hence a minus 2 for balancing it out
    check_layer = TC_layer - 2
    while flag == 0 :
        if check_layer == -1:
            check_layer = 19
        check_layer_mean = tt1[check_layer]
        if check_layer_mean < 50:
            check_layer = check_layer - 1
        else:
            flag = 1
    check_layer = check_layer + 1
    return check_layer,tt1

def find_right(TC_layer,file):
    flag = 0
    file = file.loc[0:240,["TC1","TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10","TC11","TC12","TC13","TC14","TC15","TC16","TC17","TC18","TC19","TC20"]]
    tt1 = file.mean(axis=0)
    # plus one already here
    check_layer = TC_layer
    while flag == 0 :
        if check_layer == 20:
            check_layer = 0
        check_layer_mean = tt1[check_layer]
        if check_layer_mean < 50:
            check_layer = check_layer + 1
        else:
            flag = 1
    check_layer = check_layer + 1
    return check_layer

def find_opposite(TC_layer):
    # TC_layer has become the index here
    TC_layer = TC_layer - 1
    if TC_layer == 0:
        check_layer = 16
    elif TC_layer == 1:
        check_layer = 15
    elif TC_layer == 2:
        check_layer = 14
    elif TC_layer == 3:
        check_layer = 13
    elif TC_layer == 4:
        check_layer = 13
    elif TC_layer == 5:
        check_layer = 12
    elif TC_layer == 6:
        check_layer = 11
    elif TC_layer == 7:
        check_layer = 10
    elif TC_layer == 8:
        check_layer = 19
    elif TC_layer == 9:
        check_layer = 18
    elif TC_layer == 10:
        check_layer = 6
    elif TC_layer == 11:
        check_layer = 5
    elif TC_layer == 12:
        check_layer = 4
    elif TC_layer == 13:
        check_layer = 3
    elif TC_layer == 14:
        check_layer = 3
    elif TC_layer == 15:
        check_layer = 2
    elif TC_layer == 16:
        check_layer = 1
    elif TC_layer == 17:
        check_layer = 0
    elif TC_layer == 18:
        check_layer = 9
    elif TC_layer == 19:
        check_layer = 8

    check_layer = check_layer + 1
    return check_layer



orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat')
all_files = os.listdir()
random.shuffle(all_files)
count = 0
mean_dat = pd.DataFrame()
null_index = pd.DataFrame()
for file_name in all_files:
    count = count + 1
    print(count)
    file = pd.read_csv(file_name)
    file = file.dropna()
    if file.isnull().any().any():
        null_temp = 1
    else:
        null_temp = 0
        
    if file.loc[0,'layer'] == 2 :
        TC_layer = int(np.mean(file.loc[:,'TC_layer']))
        tt1 = 'TC' + str(TC_layer)
        n=180
        L1 = file.loc[n:245,tt1]
        ML = file.loc[n:245,'M.level']
        CS = file.loc[n:245,'C.speed']
        CP = file.loc[n:245,'C.percent']
        MW = file.loc[n:245,'M.width']

        TC_layer_opp = find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        LO1 = file.loc[n:245,tt1]

        print('triggering layer : ', TC_layer, 'index : ',TC_layer - 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(2,2,2)
        plt.plot(range(len(LO1)),pp.scale(LO1),range(len(LO1)),pp.scale(L1))
        plt.legend(['opposite','self','difference'])
        plt.grid(1)
        title_string = 'mean : ' + str(np.mean(LO1)) + ' std : ' + str(np.std(LO1))
        plt.title(title_string)

        plt.subplot(2,2,1)
        plt.plot(range(len(ML)),ML+70,range(len(ML)),L1)
        plt.legend(['mold level + 70',' layer 1'])
        plt.grid(1)
        plt.title('Mold Level')

        plt.subplot(2,2,3)
        plt.scatter(pp.scale(L1),pp.scale(ML))
        plt.xlabel('Layer 1')
        plt.grid(1)
        plt.title('L1 minus ML')

        plt.subplot(2,2,4)
        plt.scatter(pp.scale(L1),pp.scale(LO1))
        plt.xlabel('Layer 1')
        plt.grid(1)
        plt.title('L1 minus Opposite')
        plt.suptitle(file_name)
        plt.show()
