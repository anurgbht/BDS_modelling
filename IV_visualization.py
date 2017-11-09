import os
import numpy as np
import pandas as pd
import random
import pylab as plt

##################################################################################################
##################################################################################################
##################################################################################################

def make_woe(x_all,x0,x1):
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
##    bins = np.percentile(x_all,[10,25,50,75,90])
    bins = np.linspace(min([min(x0),min(x1)]), max([max(x0),max(x1)]), 20)
    n, bins, patches = plt.hist([x0,x1],bins = bins)
    plt.legend(['zeros','ones'])
    plt.grid(1)

    zeros = list(n[0])
    ones = list(n[1])
    zeros = zeros/sum(zeros)
    ones = ones/sum(ones)
    woe = np.log(ones/zeros)
    mult = ones-zeros
    woe[np.isinf(woe)] = 0
    iv = sum([x*y for x,y in zip(mult,woe)])
    tt = []
    for j in range(len(bins)-1):
##        tt.append((bins[j]+bins[j+1])/2)
        tt.append(bins[j])

    plt.subplot(2,1,2)
    plt.plot(tt,woe,'*-')
    plt.xlim([min(x0),max(x0)])
    plt.grid(1)

    plt.suptitle('variable name : ' + x_all.name + ' IV of the variable = ' + str(iv))
    
    plt.savefig(x_all.name+'_plot.png')
    plt.close()
##    plt.show()
##

def make_boxplot(x_all,x0,x1):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.boxplot(x0.values)
    plt.ylim([min(x_all),max(x_all)])
    plt.grid(1)
    plt.title('Zeros Boxplot')

    plt.subplot(1,2,2)
    plt.boxplot(x1.values)
    plt.ylim([min(x_all),max(x_all)])
    plt.grid(1)
    plt.title('Ones Boxplot')

    plt.suptitle(x_all.name + ' : Boxplot comparison')
    plt.savefig(x_all.name +'_plot.png')
    plt.close()
##    plt.show()

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data')

dat = pd.read_csv('for_level_2_05_10.csv')

X_all = dat.iloc[:,3:-1]
Y_all = dat.iloc[:,-1]

##x_temp1 = X_all.loc[Y_all == 1,:]
##x_temp0 = X_all.loc[Y_all == 0,:]

x_temp1 = X_all.loc[[x==1 for x in list(Y_all)],:]
x_temp0 = X_all.loc[[x==0 for x in list(Y_all)],:]

for i in range(x_temp1.shape[1]):
    x_all = X_all.iloc[:,i]
    x1 = x_temp1.iloc[:,i].dropna()
    x0 = x_temp0.iloc[:,i].dropna()

    make_woe(x_all,x0,x1)
