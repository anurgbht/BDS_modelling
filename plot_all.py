import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    return check_layer

def find_right(TC_layer,file):
    flag = 0
    file = file.loc[:,["TC1","TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10","TC11","TC12","TC13","TC14","TC15","TC16","TC17","TC18","TC19","TC20"]]
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
os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/alarm files/Casrer3 Apr-Sep15 csv')
all_files = os.listdir()
tick_x = np.linspace(220,250,4)
temp=pd.DataFrame()
count = 0
for file_name in all_files:
    if (file_name != 'file_modifications.py') & (file_name != 'new_dat') & (file_name != 'plots'):
        count = count + 1
        print(count)
        j = file_name.split('_')
        year_code = j[0][0]
        if year_code == 'G':
            year = 2016
        elif year_code == 'F':
            year = 2015
        month_code = j[0][1]
        if month_code == 'a':
            month = '10'
        elif month_code == 'b':
            month = '11'
        elif month_code == 'c':
            month = '12'
        else:
            month = month_code
        # date 2,3
        date = (int(j[0][2]))*10 + int(j[0][3])
        hour = (int(j[0][4]))*10 + int(j[0][5])
        minute = (int(j[0][6]))*10 + int(j[0][7])

        BV = j[1]
        reason = j[2]
        layer = j[3]
        TC_layer = j[4].split('.')[0]
        TC_no = 20*(int(layer)-1) + int(TC_layer)

        TC_layer = int(TC_layer) + 1

        file = pd.read_csv(file_name)
        flag = 0
        tt1 = 'TC' + str(TC_layer)
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)
        tt3 = 'TC' + str(TC_layer + 40)
        tt4 = 'TC' + str(TC_layer + 60)
        layer = [tt1,tt2,tt3,tt4]
        
        y1 = file.loc[:,tt1]
        y2 = file.loc[:,tt2]
        y3 = file.loc[:,tt3]
        y4 = file.loc[:,tt4]
        x = np.linspace(0,300,len(y1))
        plt.clf()
        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(4,1,1)
        plt.plot(x,y1,x,y2,x,y3,x,y4)
        plt.title('Trigger ThermoCouple')
        plt.xticks([x for x in tick_x])
        plt.legend(['Level 1','Level 2','Level 3','Level 4'])
        plt.grid(1)

        TC_layer_left = find_left(TC_layer,file)
        
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

        TC_layer_right = find_right(TC_layer,file)
        
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

        print(j)
        sup_title = 'file name : ' + file_name + ' trigger TC : ' + str(TC_layer) + ' layer : ' + j[3] + ' Carbon % : ' + str(np.mean(file.loc[:,'C.percent'])) + ' Mold width : ' + str(np.mean(file.loc[:,'M.width']))

        plt.suptitle(sup_title)

        plot_save = 'D://OneDrive - Tata Insights and Quants, A division of Tata Industries//Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//second iteration//caster C old plots//' + file_name.split('.')[0] + '.png'
        plt.savefig(plot_save)
        plt.close()
##        plt.show()

##        temp_string = [file_name,year,month,date,BV,reason,layer,TC_layer,TC_no,j[6]]
##        print(temp_string)
##        temp = temp.append(pd.DataFrame(temp_string).transpose())
