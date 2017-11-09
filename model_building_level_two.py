import os
import numpy as np
import pandas as pd
import BDS_module_10_10_17 as bds

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data')

dat0 = pd.read_csv('for_ANN_0.csv')
dat1 = pd.read_csv('for_ANN_1.csv')
dat11 = pd.read_csv('for_ANN_11.csv')
dat = pd.concat([dat0,dat1,dat11],axis=0).reset_index().drop('index',1)
print(dat.shape)
X_all = dat.iloc[:,:-1]
Y_all = dat.iloc[:,-1]
####clf_logit,missed_files = bds.prelim_logit(X_all,X_all,Y_all,Y_all,file_names,0.2,1)
####clf_gbm,missed_files = bds.prelim_gbm(X_all,Y_all,file_names,0.2,1)
clf,missed_files = bds.prelim_keras(X_all,Y_all,0.2,1)
####clf_rf = bds.prelim_RF(X_all,Y_all,0.2)
####clf_svm = bds.prelim_svm(X_all,Y_all,0.2)
##clf,missed_files = bds.prelim_knn(X_all,Y_all,file_names,0.2,1)
##
##print(missed_files)
####input()


