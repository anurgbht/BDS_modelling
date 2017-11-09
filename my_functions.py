import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn import preprocessing as pp
import random
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from keras.models import Sequential
from keras.layers import Dense,Dropout
import pydotplus
from sklearn.svm import SVC


def train_test_split_my(x_data, y_data, test_size,random_state):
    rs = int(np.round_(len(y_data)*test_size))
    tt = range(len(y_data))
    test_index = random.sample(tt,rs)
    train_index = pd.DataFrame()
    for i in tt:
        if i not in test_index:
            train_index = train_index.append([i])
            test_index = [x for x in test_index]
    test_index = pd.DataFrame(test_index)
    x_data = pd.DataFrame(pp.scale(x_data))
    y_data = pd.DataFrame(y_data)
    X_train = x_data.iloc[train_index.iloc[:,0],:]
    y_train = y_data.iloc[train_index.iloc[:,0],:]
    X_test = x_data.iloc[test_index.iloc[:,0],:]
    y_test = y_data.iloc[test_index.iloc[:,0],:]
    print(x_data.shape,y_data.shape,X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    return X_train, X_test, y_train, y_test,test_index


def my_pred_fun(clf,x_data,y_test,thresh,plot_flag=0):
    print('entering my_pred')
    pred = clf.predict_proba(x_data)
    y_pred = clf.predict(x_data)
    pred = pd.DataFrame(pred)
    print('predicted shape',pred.shape)
    pred = pred.iloc[:,-1]
    my_pred = pred
    if plot_flag == 1:
        plt.scatter(my_pred,range(0,len(my_pred)),c=y_test,s=120)
        plt.grid(1)
        plt.xticks([thresh])
        plt.show()

    for i in range(0,len(pred)):
        if pred[i] > thresh:
            my_pred[i] = 1
        else:
            my_pred[i] = 0
    print(confusion_matrix(y_test,my_pred))
    return my_pred

def my_pred_fun2(*args):
    ##clf1,clf2,x_data,y_test,thresh,plot_flag=0
    print('entering my_pred')
    pred = clf.predict_proba(x_data)
    y_pred = clf.predict(x_data)
    pred = pd.DataFrame(pred)
    print('predicted shape',pred.shape)
    pred = pred.iloc[:,-1]
    my_pred = pred
    if plot_flag == 1:
        plt.scatter(my_pred,range(0,len(my_pred)),c=y_test,s=120)
        plt.grid(1)
        plt.xticks([thresh])
        plt.show()

    for i in range(0,len(pred)):
        if pred[i] > thresh:
            my_pred[i] = 1
        else:
            my_pred[i] = 0
    print(confusion_matrix(y_test,my_pred))
    return my_pred

################################################################################
###########################MODELLING CODES HERE###############################
################################################################################

def prelim_gbm(*args):
    print('Preliminary GBM Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        proceed_flag = 1
    if proceed_flag == 1:
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
        my_pred = my_pred_fun(clf,X_test,y_test,thresh)
        print(confusion_matrix(y_test,my_pred))
        print(accuracy_score(y_test,my_pred),' : My accuracy score')
        importances = clf.feature_importances_
        for f in range(X_test.shape[1]):
            print("Feature %d (%f)" % (f + 1, importances[f]))

        return clf
    else:
        print('Something went wrong')



def prelim_sklearn(*args):
    print('Preliminary sklearn Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        proceed_flag = 1
    if proceed_flag == 1:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(12,), random_state=1)
        clf.fit(X_train, y_train)
        my_pred = my_pred_fun(clf,X_test,y_test,0.2)
        print(clf.score(X_test,y_test))
        print(confusion_matrix(y_test,my_pred))
        return clf
    else:
        print('Something went wrong')



def prelim_pybrain(*args):
    print('Preliminary PyBrain Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        proceed_flag = 1


def prelim_single_tree(*args):
    print('Preliminary single tree analysis')
    proceed_flag = 0
    if len(args) == 4:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        i = args[3]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        i=1
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 6:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        i = args[5]
        proceed_flag = 1
    if proceed_flag == 1:
        clf = tree.DecisionTreeClassifier(min_samples_leaf=25)
        clf = clf.fit(X_train,y_train)
        my_pred = my_pred_fun(clf,X_test,y_test,thresh)
        print(confusion_matrix(y_test,my_pred))
        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        pdf_name = "temp_vis_" + str(i) + ".pdf"
        graph.write_pdf(pdf_name)
        print(confusion_matrix(y_test,my_pred))
        return clf
    else:
        for ii in range(7):
            print('Something went wrong')


def prelim_RF(*args):
    print('Preliminary Random Forest Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        proceed_flag = 1
    if proceed_flag == 1:
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
        y_pred_my = my_pred_fun(clf,X_test,y_test,thresh)
        print(confusion_matrix(y_test,y_pred_my))
        print(accuracy_score(y_test,y_pred_my),' : My accuracy score')
    ##    importances = clf.feature_importances_
    ##    for f in range(X_test.shape[1]):
    ##        print("Feature %d (%f)" % (f + 1, importances[f]))
        return clf
    else:
        print('Something went wrong')


def prelim_logit(*args):
    print('Preliminary Logistic Analysis')
    print(len(args))
    proceed_flag = 0
    if len(args) == 4:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        plot_flag = args[3]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 6:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        plot_flag = args[5]
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        plot_flag = 0
        proceed_flag = 1

    if proceed_flag == 1:
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
        my_pred = my_pred_fun(clf,X_test,y_test,thresh,plot_flag)
        print(confusion_matrix(y_test,my_pred))
        print(accuracy_score(y_test,my_pred),' : My accuracy score')
        return clf
    else:
        print('Something went wrong')

def prelim_keras(*args):
    print('Preliminary Keras Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        proceed_flag = 1
    if proceed_flag == 1:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = Sequential()
        clf.add(Dense(35, input_dim=21, init='uniform', activation='relu'))
##        clf.add(Dropout(0.3))
        clf.add(Dense(16, activation='sigmoid'))
##        clf.add(Dropout(0.3))
        clf.add(Dense(1, activation='sigmoid'))
        # Compile clf
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the clf
        clf.fit(X_train, y_train,nb_epoch=150, batch_size=32,  verbose=0)

        # calculate predictions
        X_test = np.array(X_test)
        pred = clf.predict_proba(X_test)
        my_pred = my_pred_fun(clf,X_test,y_test,thresh)

        return clf
    else:
        print('Something went wrong')

def prelim_svm(*args):
    print('Preliminary SVM Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3,random_state=42)
        proceed_flag = 1
    elif len(args) == 5:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        thresh = args[4]
        proceed_flag = 1
    if proceed_flag == 1:
        clf = SVC(probability=True)
        clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        y_pred = pd.DataFrame(clf.predict(X_test))
        print(y_test.shape,' : Test set size')
        print(y_test.sum(),' : Number of Ones in Test')
        print(score,' : Test Score')
        score = clf.score(X_train,y_train)
        print(score,' : Train Score')
        print(confusion_matrix(y_test,y_pred))
        y_pred_my = my_pred_fun(clf,X_test,y_test,0.1)
        print(confusion_matrix(y_test,y_pred_my))
        print(accuracy_score(y_test,y_pred_my),' : My accuracy score')
    ##    importances = clf.feature_importances_
    ##    for f in range(X_test.shape[1]):
    ##        print("Feature %d (%f)" % (f + 1, importances[f]))
        return clf
    else:
        print('Something went wrong')

################################################################################
###########################THERMOCOUPLE FINDING HERE############################
################################################################################

def find_active(m_width):
    # this takes full mold width as input
    # do not give it half width
    half_width = m_width/2
    if half_width >= 770:
        therm_list = [x+1 for x in range(20)]
    elif ((half_width>=670) & (half_width < 770)):
        therm_list = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20]
    elif ((half_width>=570) & (half_width < 670)):
        therm_list = [2,3,4,5,6,7,9,10,12,13,14,15,16,17,19,20]
    elif ((half_width>=470) & (half_width < 570)):
        therm_list = [2,3,4,5,6,9,10,12,13,14,15,16,19,20]
    elif ((half_width>=370) & (half_width < 470)):
        therm_list = [3,4,5,6,9,10,13,14,15,16,19,20]
    else:
        therm_list = -1
    return therm_list

def find_left(m_width,TC_layer):
    therm_list = find_active(m_width)
    try:
        temp = therm_list.index(TC_layer)
        if temp == 0:
            check_index = therm_list[-1]
        else:
            check_index = therm_list[temp-1]
    except:
        [print('trigger thermocouple not found in the active list') for x in range(7)]
        check_index = -1
    return check_index
    
def find_right(m_width,TC_layer):
    therm_list = find_active(m_width)
    try:
        temp = therm_list.index(TC_layer)
        if temp == (len(therm_list)-1):
            check_index = therm_list[0]
        else:
            check_index = therm_list[temp+1]
    except:
        [print('trigger thermocouple not found in the active list') for x in range(7)]
        check_index = -1
    return check_index


def find_left2(TC_layer,file):
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

def find_right2(TC_layer,file):
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

################################################################################
###########################FEATURE GENERATION HERE##############################
################################################################################

def slope_n(tt,n):
    temp = tt.iloc[-13:].reset_index().iloc[:,-1]
    max_t = temp.idxmax()
    m = (13 - max_t) + n
    temp2 = (max(temp) - tt.iloc[-m])
    return temp2

def find_drop(l2):
    temp = max(l2.iloc[-7:]) - l2.iloc[-1]
    return temp

def last_n(tt,n):
    temp = tt.iloc[-13:].reset_index().iloc[:,-1]
    temp1 = max(temp)
    if n > 10:
        temp2 = np.mean(tt[-(n+5):-(n-5)])
    else:
        temp2 = np.mean(tt[-(n+2):-(n-2)])
    temp3 = temp1-temp2
    return temp3
    
## what is the magnitude of the cross over n seconds before the peak
## it should be negative for lower values of n and positive for higher values
def sign_present(tt1,tt2,n):
    temp = tt1.iloc[-13:].reset_index().iloc[:,-1]
    # this still gives me index as in the original one
    max_t = temp.idxmax()
    m = (13 - max_t) + n
    temp1 = tt1.iloc[-m] - tt2.iloc[-m]
    return temp1

def find_l1_peak_slope(l1,cs):
    temp=0
    n=0
    while temp<100:
        if n<len(cs)-7:
            n=n+1
            temp = temp + (50/3)*(cs.iloc[-n])
        else:
            n=len(cs)-7
            temp = 101
    temp2 = l1
    temp3 = temp2.iloc[-n] - temp2.iloc[-(n+7)]
    return temp3

def find_l1_peak_ratio(l1,cs):
    temp=0
    n=0
    while temp<100:
        if n<len(cs)-7:
            n=n+1
            temp = temp + (50/3)*(cs.iloc[-n])
        else:
            n=len(cs)-7
            temp = 101
    temp2 = l1
    temp3 = (temp2.iloc[-n])/np.mean(temp2.iloc[-(n+15):-(n+7)])
    return temp3

def cs_change(cs):
    temp = cs.iloc[-15:].reset_index().iloc[:,-1]
    temp2 = np.std(temp)
    return temp2

def get_kinks(l2):
    temp = l2.iloc[-15:].reset_index().iloc[:,-1]
    temp2 = pd.DataFrame()
    for i in range(len(temp)-1):
        temp2 = temp2.append([temp.iloc[i+1] - temp.iloc[i]])
    temp3 = abs(temp2)
    temp4 = min(temp3.iloc[:,0])
    return temp4

def first_derivative(l2):
    temp = l2.iloc[-10:].reset_index().iloc[:,-1]
    temp2 = pd.DataFrame()
    for i in range(len(temp)-1):
        temp2 = temp2.append([temp.iloc[i+1] - temp.iloc[i]])
    return max(temp2.iloc[:,0])-min(temp2.iloc[:,0])

def second_derivative(l2):
    temp = l2.iloc[-10:].reset_index().iloc[:,-1]
    temp2 = pd.DataFrame()
    temp3 = pd.DataFrame()
    for i in range(len(temp)-1):
        temp2 = temp2.append([temp.iloc[i+1] - temp.iloc[i]])
    for i in range(len(temp2)-1):
        temp3 = temp3.append([temp2.iloc[i+1] - temp2.iloc[i]])
##    print(max(abs(temp3.iloc[:,0])))
    return max(abs(temp3.iloc[:,0]))

def bin_crossover(l1,l2):
    temp = l2.iloc[-13:].reset_index().iloc[:,-1]
    max_t = temp.idxmax()
    m = (13 - max_t)
    temp2 = l2.iloc[-m] - l1.iloc[-m]
    return temp2

def peak_or_not(l2):
    t1 = max(l2)
    t2 = np.mean(l2)
    t3 = np.std(l2)
    if ((t1-t2)/t3) > 2.5:
        return 1
    else:
        return 0

def peak_diff(l1,l2):
    temp1 = l1.iloc[-25:].reset_index().iloc[:,-1]
    max_t1 = temp1.idxmax()
    temp2 = l2.iloc[-25:].reset_index().iloc[:,-1]
    max_t2 = temp2.idxmax()
##    print('max1-max2',max_t1-max_t2,max_t1,max_t2,np.std(temp1),np.std(temp2))
##
##    plt.subplot(2,1,1)
##    plt.plot(range(len(l1)),l1,range(len(l2)),l2)
##    plt.legend(['l1','l2'])
##
##    plt.subplot(2,1,2)
##    plt.plot(range(len(temp1)),temp1,range(len(temp2)),temp2)
##    plt.legend(['temp1','temp2'])
##    plt.show()
    if (np.std(temp1) < 4) or (np.std(temp2) < 4):
        tt = -99
    else:
        tt1 = max_t2-max_t1
        if tt1 < 0:
            tt = -99
        else:
            tt = tt1

    return tt

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def make_x(L1,L2,ML,CP,CS,MW,tag,file_name):
    x_temp = pd.DataFrame()
    y_temp = pd.DataFrame()
    index_temp = pd.DataFrame()
    n = 61
    k = 4
    for i in range(0,k):
        l1 = L1[i*n:(i+1)*n].reset_index().iloc[:,-1]
        l2 = L2[i*n:(i+1)*n].reset_index().iloc[:,-1]
        ml = ML[i*n:(i+1)*n].reset_index().iloc[:,-1]
        cs = CS[i*n:(i+1)*n].reset_index().iloc[:,-1]
        cp = CP[i*n:(i+1)*n].reset_index().iloc[:,-1]
        mw = MW[i*n:(i+1)*n].reset_index().iloc[:,-1]

        x_temp = x_temp.append([[
            slope_n(l2,3),
            slope_n(l2,5),
            slope_n(l2,7),
            sign_present(l1,l2,2),
            sign_present(l1,l2,5),
            sign_present(l1,l2,7),
            last_n(l1,5),
            last_n(l2,5),
            last_n(l1,32),
            last_n(l2,32),
            np.mean(cp),
            np.std(l1.iloc[-12:]),
            np.std(ml.iloc[-12:]),
            find_l1_peak_slope(l1,cs),
            find_l1_peak_ratio(l1,cs),
            find_drop(l1),
            find_drop(l2),
            cs_change(cs),
            get_kinks(l2),
            first_derivative(l2)
                                ]])
        index_temp = index_temp.append([file_name])

        if i  == (k-1) and tag == 'True':
            y_temp = y_temp.append([1])
        else:
            y_temp = y_temp.append([0])
    return x_temp,y_temp,index_temp


def make_cont_x(L1,L2,ML,CP,CS,MW):
    x_temp = pd.DataFrame()

    for i in range(0,len(L1)-63):
        l1 = L1[i:i+61].reset_index().iloc[:,-1]
        l2 = L2[i:i+61].reset_index().iloc[:,-1]
        ml = ML[i:i+61].reset_index().iloc[:,-1]
        cs = CS[i:i+61].reset_index().iloc[:,-1]
        cp = CP[i:i+61].reset_index().iloc[:,-1]
        mw = MW[i:i+61].reset_index().iloc[:,-1]

        x_temp = x_temp.append([[
            slope_n(l2,3),
            slope_n(l2,5),
            slope_n(l2,7),
            sign_present(l1,l2,2),
            sign_present(l1,l2,5),
            sign_present(l1,l2,7),
            last_n(l1,5),
            last_n(l2,5),
            last_n(l1,32),
            last_n(l2,32),
            np.mean(cp),
            np.std(l1.iloc[-12:]),
            np.std(ml.iloc[-12:]),
            find_l1_peak_slope(l1,cs),
            find_l1_peak_ratio(l1,cs),
            find_drop(l1),
            find_drop(l2),
            cs_change(cs),
            get_kinks(l2),
            first_derivative(l2)
                                ]])
##    print(x_temp.shape, " : shape of the returned x_trigger")
    return x_temp

def make_one_x(L1,L2,ML,CP,CS):
    #print('Making continuous X')
    x_temp = pd.DataFrame()
    
    x_temp = x_temp.append([[
            slope_n(L2,3),
            slope_n(L2,5),
            slope_n(L2,7),
            sign_present(L1,L2,2),
            sign_present(L1,L2,5),
            sign_present(L1,L2,7),
            last_n(L1,5),
            last_n(L2,5),
            last_n(L1,32),
            last_n(L2,32),
            CP,
            np.std(L1.iloc[-12:]),
            np.std(ML.iloc[-12:]),
            find_l1_peak_slope(L1,CS),
            find_l1_peak_ratio(L1,CS),
            find_drop(L1),
            find_drop(L2),
            cs_change(CS),
            get_kinks(L2),
            first_derivative(L2)
        ]])

    return x_temp



# for tree based classifier
def make_bin_x(L1,L2,ML,CP,CS,MW):
    print('Making Binary X')
    x_temp = pd.DataFrame()
    for i in range(0,238):
        print(x_temp.shape)
        # print(i)
        l1 = L1[i:i+61].reset_index().iloc[:,-1]
        l2 = L2[i:i+61].reset_index().iloc[:,-1]
        ml = ML[i:i+61].reset_index().iloc[:,-1]
        cs = CS[i:i+61].reset_index().iloc[:,-1]
        cp = CP[i:i+61].reset_index().iloc[:,-1]
        mw = MW[i:i+61].reset_index().iloc[:,-1]

        x_temp = x_temp.append([[
            bin_crossover(l1,l2),
            
                                ]])

    return x_temp


#####################################################################################

def calc_WOE(X,y):

    temp2 = pd.DataFrame()
    tg = sum(y)
    tb = len(y) - tg
    for i in set(X):
        temp = y[X == i]
        ttg = sum(temp)/tg
        ttb = (len(temp)-sum(temp))/tb
        ttl = np.log((ttg/ttb))
        temp2 = temp2.append([[i,ttl]])
    print(temp2)
    return temp2

#####################################################################################

def my_plot(x):
    plt.plot(range(len(x)),x)
    plt.grid(1)
    plt.show()


#####################################################################################
#####################################################################################
#####################################################################################

##L = pd.read_csv("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat/F1041807_103_01_2_13_CASTER-C_False_JAN'15_new.csv")
##L2 = L.TC34
##L1 = L.TC14
##for i in range(4):
##    l1 = L1.iloc[i:61*i + 61]
##    l2 = L2.iloc[i:61*i + 61]
##    peak_diff(l1,l2)
