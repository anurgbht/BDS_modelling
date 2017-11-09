import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn import preprocessing as pp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
import shutil
from keras.models import Sequential
from keras.layers import Dense,Dropout
import pydotplus
from sklearn import tree

def include_tree(row):
    x2 = row[1]
    x6 = row[5]
    x7 = row[6]
    x16 = row[15]

    if (x2<=0.1793):
        if (x7<=-0.7124):
            if (x6<=0.1891):
                tt = -0.4546
            elif(x6>0.1891):
                tt = 2.6383
        elif(x7>-0.7124):
            if(x6<=-0.3407):
                tt = -4.8598
            elif(x6>-0.3407):
                tt = -4.8598
    elif(x2>0.1793):
        if(x6<=-0.5404):
            tt = 0.2385
        elif(x6>-0.5404):
            if(x7<=-0.256):
                if(x16<=0.1502):
                    tt = 1.8404
                elif(x16>0.1502):
                    tt = 4.0298
            elif(x7>-0.256):
                tt = 0.6753
    return tt

###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_2_3_17.csv')
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

X_all_left = pd.DataFrame(pp.scale(temp.iloc[:,0:17]))
X_all_trigger = pd.DataFrame(pp.scale(temp.iloc[:,17:34]))
X_all_right = pd.DataFrame(pp.scale(temp.iloc[:,34:51]))
Y_all_clean = temp.iloc[:,51]
index_file_clean = temp.iloc[:,52]

print(X_all_trigger.shape,' : Size of all X trigger')
print(X_all_left.shape,' : Size of all X left')
print(X_all_right.shape,' : Size of all X right')
print(Y_all_clean.shape,' : Size of all Y')
print(index_file_clean.shape,' : Shape of index file')

X_all_trigger.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17"]
X_all_left.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17"]
X_all_right.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17"]

temp = X_all_trigger.apply(include_tree,axis=1)

X_logit = pd.concat([X_all_trigger,temp],axis=1)

all_index = list(range(0,len(Y_all_clean)))
random.shuffle(all_index)

# K-cross validation for K=5, the code can be scaled easily
temp = round(len(Y_all_clean)/5)

index1 = all_index[0:temp]
index2 = all_index[temp:2*temp]
index3 = all_index[2*temp:3*temp]
index4 = all_index[3*temp:4*temp]
index5 = all_index[4*temp:]

index_file1 = index_file_clean[index1]
index_file2 = index_file_clean[index2]
index_file3 = index_file_clean[index3]
index_file4 = index_file_clean[index4]
index_file5 = index_file_clean[index5]

X_all1 = X_logit.iloc[index1,:]
Y_all1= Y_all_clean[index1]

X_all2 = X_logit.iloc[index2,:]
Y_all2= Y_all_clean[index2]

X_all3 = X_logit.iloc[index3,:]
Y_all3= Y_all_clean[index3]

X_all4 = X_logit.iloc[index4,:]
Y_all4= Y_all_clean[index4]

X_all5 = X_logit.iloc[index5,:]
Y_all5= Y_all_clean[index5]

X_test1 = X_all1
Y_test1 = Y_all1
X_train1 = pd.concat([X_all2,X_all3,X_all4,X_all5],axis=0)
Y_train1 = pd.concat([Y_all2,Y_all3,Y_all4,Y_all5],axis=0)

X_test2 = X_all2
Y_test2 = Y_all2
X_train2 = pd.concat([X_all1,X_all3,X_all4,X_all5],axis=0)
Y_train2 = pd.concat([Y_all1,Y_all3,Y_all4,Y_all5],axis=0)

X_test3 = X_all3
Y_test3 = Y_all3
X_train3 = pd.concat([X_all2,X_all1,X_all4,X_all5],axis=0)
Y_train3 = pd.concat([Y_all2,Y_all1,Y_all4,Y_all5],axis=0)

X_test4 = X_all4
Y_test4 = Y_all4
X_train4 = pd.concat([X_all2,X_all3,X_all1,X_all5],axis=0)
Y_train4 = pd.concat([Y_all2,Y_all3,Y_all1,Y_all5],axis=0)

X_test5 = X_all5
Y_test5 = Y_all5
X_train5 = pd.concat([X_all2,X_all3,X_all4,X_all1],axis=0)
Y_train5 = pd.concat([Y_all2,Y_all3,Y_all4,Y_all1],axis=0)

thresh = 0.1

clf_logit1 = prelim_logit(X_train1,X_test1,Y_train1,Y_test1,thresh)
clf_logit2 = prelim_logit(X_train2,X_test2,Y_train2,Y_test2,thresh)
clf_logit3 = prelim_logit(X_train3,X_test3,Y_train3,Y_test3,thresh)
clf_logit4 = prelim_logit(X_train4,X_test4,Y_train4,Y_test4,thresh)
clf_logit5 = prelim_logit(X_train5,X_test5,Y_train5,Y_test5,thresh)

my_pred1 = my_pred_fun(clf_logit1,X_test1,Y_test1,thresh)
my_pred2 = my_pred_fun(clf_logit2,X_test2,Y_test2,thresh)
my_pred3 = my_pred_fun(clf_logit3,X_test3,Y_test3,thresh)
my_pred4 = my_pred_fun(clf_logit4,X_test4,Y_test4,thresh)
my_pred5 = my_pred_fun(clf_logit5,X_test5,Y_test5,thresh)








problem_files = pd.DataFrame()

for i in range(len(my_pred1)):
    if (my_pred1.iloc[i] == 0) & (Y_test1.iloc[i] == 1):
        print(index_file1.iloc[i])
        problem_files = problem_files.append([index_file1.iloc[i]])

for i in range(len(my_pred2)):
    if (my_pred2.iloc[i] == 0) & (Y_test2.iloc[i] == 1):
        print(index_file2.iloc[i])
        problem_files = problem_files.append([index_file2.iloc[i]])

for i in range(len(my_pred3)):
    if (my_pred3.iloc[i] == 0) & (Y_test3.iloc[i] == 1):
        print(index_file3.iloc[i])
        problem_files = problem_files.append([index_file3.iloc[i]])

for i in range(len(my_pred4)):
    if (my_pred4.iloc[i] == 0) & (Y_test4.iloc[i] == 1):
        print(index_file4.iloc[i])
        problem_files = problem_files.append([index_file4.iloc[i]])

for i in range(len(my_pred5)):
    if (my_pred5.iloc[i] == 0) & (Y_test5.iloc[i] == 1):
        print(index_file5.iloc[i])
        problem_files = problem_files.append([index_file5.iloc[i]])

if len(problem_files) > 0:

    get_plots=pd.DataFrame()

    for kk in problem_files.iloc[:,0]:
    ##    print(kk)
        get_plots = get_plots.append([str(kk).replace("_new.csv",".png")])


    os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/plot_all_TC')
    for i in get_plots.iloc[:,0]:
        try:
            shutil.copy2(i,'/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/confusing_files')
        except:
            print("Could not copy a file")
##
####    os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/confusing_files')
####    tt = os.listdir()
####    os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat')
####
####    print(len(get_plots))
####
####    for i in tt:
####        temp = i.replace(".png","_new.csv")
####        try:
####            os.remove(temp)
####        except:
####             print('file alread removed')
##else:
##    print('All turned out great after all !')
