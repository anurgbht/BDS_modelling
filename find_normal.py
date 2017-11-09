import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_opposite(TC_layer,file):
    flag = 0
    file = file.loc[0:240,["TC1","TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10","TC11","TC12","TC13","TC14","TC15","TC16","TC17","TC18","TC19","TC20"]]
    tt1 = file.mean(axis=0)
    check_layer = TC_layer + 10
    while flag == 0 :
        if check_layer > 20:
            check_layer = check_layer - 20
        check_layer_mean = tt1[check_layer-1]
        if check_layer_mean < 50:
            check_layer = check_layer - 1
        else:
            flag = 1
    print(check_layer)
    return check_layer

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat2')
all_files = os.listdir()

temp=pd.DataFrame()
count = 0
for file_name in all_files:
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
    TC_layer = j[4]
    TC_no = 20*(int(layer)-1) + int(TC_layer)

    TC_layer = int(TC_layer) + 1
    file = pd.read_csv(file_name)
    TC_layer_opp = find_opposite(TC_layer,file)

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
    l1 = y1
    plt.subplot(2,2,1)
    plt.plot(x,y1,x,y2,x,y3,x,y4)
    plt.title('Trigger ThermoCouple')
    plt.xticks([240])
    plt.legend(['Level 1','Level 2','Level 3','Level 4'],loc=3)
    plt.grid(1)


    tt1 = 'TC' + str(TC_layer_opp)
    tt2 = 'TC' + str(TC_layer_opp + 20)
    tt3 = 'TC' + str(TC_layer_opp + 40)
    tt4 = 'TC' + str(TC_layer_opp + 60)
    layer = [tt1,tt2,tt3,tt4]
    
    y1 = file.loc[:,tt1]
    y2 = file.loc[:,tt2]
    y3 = file.loc[:,tt3]
    y4 = file.loc[:,tt4]
    x = np.linspace(0,300,len(y1))
    
    plt.subplot(2,2,2)
    plt.plot(x,y1,x,y2,x,y3,x,y4)
    plt.title('Opposite ThermoCouple')
    plt.xticks([240])
    plt.legend(['Level 1','Level 2','Level 3','Level 4'],loc=3)
    plt.grid(1)

    col_scheme = pd.DataFrame(np.zeros((70,1)))
    col_scheme.iloc[-20:] = 1
    plt.subplot(2,2,3)
    plt.scatter(l1.iloc[180:250],y1.iloc[180:250],c=col_scheme)
    plt.grid(1)

    plt.subplot(2,2,4)
    plt.plot(x,file.loc[:,'M.level'],x,100*file.loc[:,'C.speed'])
    plt.title('Mold level and 100x Casting speed')
    plt.xticks([240])
    plt.legend(['Mold level','Casting speed'],loc=3)
    plt.grid(1)

    print(j)
    sup_title = 'file name : ' + j[0] + ' ' + j[5] + ' Status : ' + j[6] + ' trigger TC : ' + str(TC_layer) + ' layer : ' + j[3] + ' Carbon % : ' + str(np.mean(file.loc[:,'C.percent'])) + ' Mold width : ' + str(np.mean(file.loc[:,'M.width']))

    plt.suptitle(sup_title)

    plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plot_temp//' + j[0] + '_'+ j[1]+ '_' + j[2]+ '_' + j[3]+ '_' + j[4]+ '_' + j[5]+ '_' + j[6] + '_'+ j[7] + '.png'
    plt.savefig(plot_save)
    plt.close()
##        plt.show()

##        temp_string = [file_name,year,month,date,BV,reason,layer,TC_layer,TC_no,j[6]]
##        print(temp_string)
##        temp = temp.append(pd.DataFrame(temp_string).transpose())
