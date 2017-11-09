import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
import my_functions as mf
## MAIN HERE

outdata = pd.read_csv('D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//data_dump_smart.csv')

##ind_val=[5,6,7,8,11,12,13,14,22,23,24,25,27,28,29,30,31,32,45,46,47,48,49,52,55]

X_all = outdata.iloc[:,:-1]
##X_all = X_all.iloc[:,ind_val]
X_all = preprocessing.scale(X_all)
Y_all = pd.DataFrame(outdata.iloc[:,-1])

##clf_RF = prelim_RF(X_all,Y_all)
##clf_logit = prelim_logit(X_all,Y_all)
clf_GBM = mf.prelim_gbm(X_all,Y_all,0.2)
##clf_SVM = prelim_SVM(X_all,Y_all)
