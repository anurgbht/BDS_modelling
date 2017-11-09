import os
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
import my_functions as mf
    
###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

orginal_path = os.getcwd()

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_3_3_17.csv')
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

##thresh = 0.05
##clf_keras_trigger,my_pred,y_test = prelim_keras(X_all_trigger,Y_all_clean,thresh)
##clf_keras_left,my_pred,y_test = prelim_keras(X_all_left,Y_all_clean,thresh)
##clf_keras_right,my_pred,y_test = prelim_keras(X_all_right,Y_all_clean,thresh)

clf,X_train,X_test,y_test,y_train,test_index,my_pred = mf.prelim_single_tree(X_all_trigger,Y_all_clean,0.1)
