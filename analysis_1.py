import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def my_pred(clf,x_data,y_test,thresh):
    print('entering my_pred')
    pred = clf.predict_proba(x_data)
    y_pred = clf.predict(x_data)
    pred = pd.DataFrame(pred)
    print('predicted shape',pred.shape)
    pred = pred.iloc[:,-1]
    my_pred = pred
    plt.scatter(my_pred,range(0,len(my_pred)),c=y_test,s=120)
    plt.grid(1)
    plt.xticks([0.1])
    plt.show()

    for i in range(0,len(pred)):
        if pred[i] > thresh:
            my_pred[i] = 1
        else:
            my_pred[i] = 0
    return my_pred

def last_nm(tt,n,m):
    temp1 = tt.iloc[-n:]
    temp2 = tt.iloc[-m:]
    temp3 = np.mean(temp1)/np.mean(temp2)
    return temp3

def slope_n(tt,n):
    temp2 = (np.mean(tt.iloc[-3:]) - np.mean(tt.iloc[-(n+2):-(n-1)]))/n
    return temp2

def sign_present(tt1,tt2):
    temp1 = np.mean(tt1.iloc[-3:]) - np.mean(tt2.iloc[-3:])
    temp2 = np.sign(temp1)
    return temp2


def make_x(L1,L2,L3,L4,ML,CP,CS,tag):
    x_temp = pd.DataFrame()
    y_temp = pd.DataFrame()
    n=81
    for i in range(0,3):
##        print(i)
        tt1 = L1[i*n:(i+1)*n]
        tt2 = L2[i*n:(i+1)*n]
        tt3 = ML[i*n:(i+1)*n]
        tt4 = CS[i*n:(i+1)*n]
        tt5 = CP[i*n:(i+1)*n]

        x_temp = x_temp.append([[
                                np.sqrt(np.mean(np.square(tt1))),
                                np.max(tt1),np.mean(tt1),np.min(tt1),np.std(tt1),
                                last_nm(tt1,2,12),last_nm(tt1,5,12),last_nm(tt1,7,12),
                                last_nm(tt1,2,32),last_nm(tt1,5,32),last_nm(tt1,7,32),
                                slope_n(tt1,10),slope_n(tt1,5),slope_n(tt1,15),slope_n(tt1,20),
                                np.sqrt(np.mean(np.square(tt2))),
                                np.max(tt2),np.mean(tt2),np.min(tt2),np.std(tt2),
                                last_nm(tt2,2,12),last_nm(tt2,5,12),last_nm(tt2,7,12),
                                last_nm(tt2,2,32),last_nm(tt2,5,32),last_nm(tt2,7,32),
                                slope_n(tt2,10),slope_n(tt2,5),slope_n(tt2,15),slope_n(tt2,20),
                                np.sqrt(np.mean(np.square(tt3))),
                                np.max(tt3),np.mean(tt3),np.min(tt3),np.std(tt3),
                                last_nm(tt3,2,12),last_nm(tt3,5,12),last_nm(tt3,7,12),
                                last_nm(tt3,2,32),last_nm(tt3,5,32),last_nm(tt3,7,32),
                                slope_n(tt3,10),slope_n(tt3,5),slope_n(tt3,15),slope_n(tt3,20),
                                np.sqrt(np.mean(np.square(tt4))),
                                np.max(tt4),np.mean(tt4),np.min(tt4),np.std(tt4),
                                last_nm(tt4,2,12),last_nm(tt4,5,12),last_nm(tt4,7,12),
                                last_nm(tt4,2,32),last_nm(tt4,5,32),last_nm(tt4,7,32),
                                slope_n(tt4,10),slope_n(tt4,5),slope_n(tt4,15),slope_n(tt4,20),
                                np.mean(tt5),
                                sign_present(tt1,tt2)
                                ]])
        if i  == 2 and tag == 'True':
            y_temp = y_temp.append([1])
        else:
            y_temp = y_temp.append([0])
    return x_temp,y_temp

def prelim_RF(x_data,y_data):
    print('Preliminary Random Forest Analysis')
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,random_state=42)
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    y_pred = pd.DataFrame(clf.predict(X_test))
    print(y_test.shape,' : Test set size')
    print(y_test.sum(),' : Number of Ones in Test')
    print(score,' : Test Score')
    score = clf.score(X_train,y_train)
    print(score,' : Train Score')
    print(confusion_matrix(y_test,y_pred))
    y_pred_my = my_pred(clf,X_test,y_test,0.1)
    print(confusion_matrix(y_test,y_pred_my))
    print(accuracy_score(y_test,y_pred_my),' : My accuracy score')
    importances = clf.feature_importances_
    for f in range(X_test.shape[1]):
        print("Feature %d (%f)" % (f + 1, importances[f]))
    return clf

def prelim_logit(x_data,y_data):
    print('Preliminary Logistic Analysis')
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    y_pred = pd.DataFrame(clf.predict(X_test))
    print(y_test.shape,' : Test set size')
    print(y_test.sum(),' : Number of Ones in Test')
    print(score,' : Test Score')
    score = clf.score(X_train,y_train)
    print(score,' : Train Score')
    print(confusion_matrix(y_test,y_pred))
    y_pred_my = my_pred(clf,X_test,y_test,0.1)
    print(confusion_matrix(y_test,y_pred_my))
    print(accuracy_score(y_test,y_pred_my),' : My accuracy score')
##    importances = clf.feature_importances_
##    for f in range(X_test.shape[1]):
##        print("Feature %d (%f)" % (f + 1, importances[f]))
    return clf


def prelim_GBM(x_data,y_data):
    print('Preliminary GBM Analysis')
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,random_state=42)
    clf = GradientBoostingClassifier()
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    y_pred = pd.DataFrame(clf.predict(X_test))
    print(y_test.shape,' : Test set size')
    print(y_test.sum(),' : Number of Ones in Test')
    print(score,' : Test Score')
    score = clf.score(X_train,y_train)
    print(score,' : Train Score')
    print(confusion_matrix(y_test,y_pred))
    y_pred_my = my_pred(clf,X_test,y_test,0.1)
    print(confusion_matrix(y_test,y_pred_my))
    print(accuracy_score(y_test,y_pred_my),' : My accuracy score')
##    importances = clf.feature_importances_
##    for f in range(X_test.shape[1]):
##        print("Feature %d (%f)" % (f + 1, importances[f]))
    return clf


orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/new_dat2')
all_files = os.listdir()

temp=pd.DataFrame()
count = 0
X_all = pd.DataFrame()
Y_all = pd.DataFrame()
for file_name in all_files:
    if (file_name != 'file_modifications.py') & (file_name != 'new_dat') & (file_name != 'plots'):
        count = count + 1
        print(count)
        file = pd.read_csv(file_name)
        if file.loc[0,'layer'] == 2:
            TC_layer = int(np.mean(file.loc[:,'TC_layer']))
            tt1 = 'TC' + str(TC_layer)
            tt2 = 'TC' + str(TC_layer + 20)
            tt3 = 'TC' + str(TC_layer + 40)
            tt4 = 'TC' + str(TC_layer + 60)
            layer = [tt1,tt2,tt3,tt4]

            L1 = file.loc[:,tt1]
            L2 = file.loc[:,tt2]
            L3 = file.loc[:,tt3]
            L4 = file.loc[:,tt4]
            ML = file.loc[:,'M.level']
            CS = file.loc[:,'C.speed']
            CP = file.loc[:,'C.percent']

            tx,ty = make_x(L1,L2,L3,L4,ML,CP,CS,file_name.split('_')[6])
            X_all = X_all.append(tx)
            Y_all = Y_all.append(ty)

##plt.scatter(X_all.iloc[:,0],X_all.iloc[:,1],c=Y_all,s=140)
##plt.xlabel('RMS')
##plt.ylabel('Max')
##plt.grid(1)
##plt.show()

print(X_all.shape,' : Size of all X')
print(Y_all.shape,' : Size of all Y')

outdata = X_all
outdata['y_label'] = Y_all
outdata = outdata.dropna()
##outdata.to_csv('all_data.csv')

X_all = outdata.iloc[:,:-1]
X_all = preprocessing.scale(X_all)
Y_all = pd.DataFrame(outdata.iloc[:,-1])

##IV_table = find_IV(X_all,Y_all,2)

##clf_RF = prelim_RF(X_all,Y_all)
##clf_logit = prelim_logit(X_all,Y_all)
clf_GBM = prelim_GBM(X_all,Y_all)
