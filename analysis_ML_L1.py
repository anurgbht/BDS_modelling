import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,linear_model
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from time import sleep


orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat')
all_files = os.listdir()

temp=pd.DataFrame()
count = 0
for file_name in all_files:
    count = count + 1
    print(count)
    file = pd.read_csv(file_name)
    if file.loc[0,'layer'] == 2:
##        print(file_name)
        TC_layer = int(np.mean(file.loc[:,'TC_layer']))
        tt1 = 'TC' + str(TC_layer)
        L1 = np.array(file.loc[0:244,tt1])
        ML = np.array(file.loc[0:244,'M.level'])
        l1 = preprocessing.scale(L1)
        ml = preprocessing.scale(ML)

        (ar,br)=polyfit(ML,L1,1)
        xr=polyval([ar,br],ML)
        
        n = len(L1)
        err = xr-L1
        rmserr=sqrt(sum((err)**2)/n)
        ybar = np.average(L1)
        sstot = sum((L1-ybar)**2)
        ssres = sum((err)**2)
        merr = np.average(err)
        stderr = np.std(err)
        rsq = 1-(ssres/sstot)

        temp = temp.append([[ar,br,rmserr,rsq,merr,stderr,np.std(ML),np.std(L1),file_name]])
##        plt.hist(err)
##        plt.show()

##        std_err1 = ((sum(err**2)/(n-2))/np.var(ML))**0.5
##        print('Linear regression using polyfit')
##        print('a=%.2f b=%.2f, ms error= %.3f, Rsquared = %.3f, mean error = %.3f, std = %.3f' % (ar,br,rmserr,rsq,merr,stderr))
##
##        tt0 = err[0:n-1]
##        tt1 = err[1:n]
##
##        slope, intercept, r_value, p_value, std_err = stats.linregress(tt0,tt1)

##        print('Linear regression using SciPy')
##        print('a=%.2f b=%.2f, ms error= %.3f, Rsquared = %.3f, mean error = %.3f, std = %.3f' % (slope,intercept,rmserr,r_value**2,merr,std_err))
##
##        sleep(3)
##
        plt.clf()
        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(ML,L1)
        plt.plot(ML,xr,c='y')
        plt.grid(1)
        plt.xlabel('Mold Level')
        plt.ylabel('Level 1')
        plt.title('a=%.2f b=%.2f, ms error= %.3f, Rsquared = %.3f, mean error = %.3f, std = %.3f' % (ar,br,rmserr,rsq,merr,stderr))
        j=file_name.split('_')
        sup_title = 'file name : ' + j[0] + ' ' + j[5] + ' Status : ' + j[6] + ' trigger TC : ' + str(TC_layer) + ' layer : ' + j[3] + ' Carbon % : ' + str(np.mean(file.loc[:,'C.percent'])) + ' Mold width : ' + str(np.mean(file.loc[:,'M.width']))
        plt.suptitle(sup_title)
        plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//plot_ML_L1_scatter//' + j[0] + '_'+ j[1]+ '_' + j[2]+ '_' + j[3]+ '_' + j[4]+ '_' + j[5]+ '_' + j[6] + '_'+ j[7] + '.png'
        plt.savefig(plot_save)
        plt.close()

temp.columns = ["slope","intercept","rmserr","rsquared","mean error","standard deviation of error","standard deviation of ML","standard deviation of L1","file name"]
temp.to_csv('error_analysis.csv',index=False)

        
