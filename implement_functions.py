import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
import pydotplus

################################################################################
################################################################################
################################################################################

def train_test_split_my(x_data, y_data, test_size):
    rs = int(np.round_(len(y_data)*test_size))
    tt = range(len(y_data))
    test_index = random.sample(tt,rs)
    train_index = [x for x in tt if x not in test_index]
    x_data = x_data.reset_index().drop('index',1)
    y_data = y_data.reset_index().drop('index',1)
    X_train = x_data.loc[train_index,:]
    y_train = y_data.loc[train_index,:]
    X_test = x_data.loc[test_index,:]
    y_test = y_data.loc[test_index,:]
    print(x_data.shape,y_data.shape,X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    return X_train, X_test, y_train, y_test,test_index

def my_pred_fun(clf,x_data,y_test,thresh,plot_flag=1):
    print('entering my_pred')
    pred = clf.predict_proba(x_data)
    y_pred = clf.predict(x_data)
    pred = pd.DataFrame(pred)
    print('predicted shape',pred.shape)
    pred = pred.iloc[:,-1]
    my_pred = pred
    if plot_flag == 1:
        print('Entering plot function')
        plt.scatter(my_pred,range(0,len(my_pred)),c=y_test,s=120)
        plt.grid(1)
        plt.xticks([0.1,0.5,thresh,0.9])
        plt.show()

    for i in range(0,len(pred)):
        if pred[i] > thresh:
            my_pred[i] = 1
        else:
            my_pred[i] = 0
    print(confusion_matrix(y_test,my_pred))
    return my_pred

################################################################################
#######################MODEL BUILDING FUNCTIONS HERE###########################
################################################################################

def prelim_gbm(*args):
    print('Preliminary GBM Analysis')
    proceed_flag = 0
    if len(args) == 3:
        x_data = args[0]
        y_data = args[1]
        thresh = args[2]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3)
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

def single_tree(*args):
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
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3)
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
        clf = LogisticRegression( max_iter=10000,verbose=1)
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
###########################THERMOCOUPLE FINDING HERE##########################
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

def give_active_list(col_no):
    if col_no == 20:
        therm_list = [x+1 for x in range(20)]
    elif col_no == 18:
        therm_list = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20]
    elif col_no == 16:
        therm_list = [2,3,4,5,6,7,9,10,12,13,14,15,16,17,19,20]
    elif col_no == 14:
        therm_list = [2,3,4,5,6,9,10,12,13,14,15,16,19,20]
    elif col_no == 12:
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
    tt = list(tt)
    temp = tt[-13:]
    max_t = temp.index(max(temp))
    m = (13 - max_t) + n
    temp2 = (max(temp) - tt[-m])
    return temp2

def find_drop(l2):
    l2 = list(l2)
    temp = max(l2[-7:]) - l2[-1]
    return temp

def last_n(tt,n):
    tt = list(tt)
    temp = tt[-13:]
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
    tt1 = list(tt1)
    tt2 = list(tt2)
    temp = tt1[-13:]
    # this still gives me index as in the original one
    max_t = temp.index(max(temp))
    m = (13 - max_t) + n
    temp1 = tt1[-m] - tt2[-m]
    return temp1

def find_l1_peak_slope(l1,l2,cs):
    l2 = list(l2)
    l2 = l2[-13:]
    l1 = list(l1)
    cs = list(cs)
    cs = cs[:(l2.index(max(l2))-13)]
    l1 = l1[:(l2.index(max(l2))-13)]
    temp=0
    n=0
    while temp<100:
        if n<len(cs)-16:
            n=n+1
            temp = temp + (50/3)*(cs[-n])
        else:
            n=len(cs)-16
            temp = 101
    temp2 = l1[-n] - l1[-(n+7)]
    return temp2

def find_l1_peak_ratio(l1,l2,cs):
    l2 = list(l2)
    l2 = l2[-13:]
    l1 = list(l1)
    cs = list(cs)
    cs = cs[:(l2.index(max(l2))-13)]
    l1 = l1[:(l2.index(max(l2))-13)]
    temp=0
    n=0
    while temp<100:
        if n<len(cs)-16:
            n=n+1
            temp = temp + (50/3)*(cs[-n])
        else:
            n=len(cs)-16
            temp = 101
    temp2 = (l1[-n])/np.mean(l1[-(n+15):-(n+7)])
    return temp2

def cs_change(cs):
    cs = list(cs)
    temp = cs[-15:]
    temp2 = np.std(temp)
    return temp2

def get_kinks(l2):
    l2 = list(l2)
    temp = l2[-15:]
    temp2 = []
    for i in range(len(temp)-1):
        temp2.append(temp[i+1] - temp[i])
    temp3 = [abs(x) for x in temp2]
    temp4 = min(temp3)
    return temp4

def first_derivative(l2):
    temp = list(l2)
    temp2 = []
    for i in range(len(temp)-1):
        temp2.append(temp[i+1] - temp[i])
    q = (max(temp2[-15:])-min(temp2[-15:]))/(max(temp2[:-15])-min(temp2[:-15]))
    temp4 = [1 if x>0 else -1 for x in np.sign(temp2)]
    count = 0
    for i in range(len(temp4)-1):
        if (temp4[i+1] != temp4[i]):
            count += 1
    return [q,count]

def time_up(l2):
    l2 = list(l2[-20:])
    max_index = l2.index(max(l2))
    count = 0
    for i in range(1,max_index):
        if l2[max_index-i]<l2[max_index-i+1]:
            count += 1
        else:
            break
    return count

def time_down(l2):
    l2 = list(l2[-20:])
    max_index = l2.index(max(l2))
    count = 0
    for i in range(max_index,len(l2)-1):
        if l2[i]>l2[i+1]:
            count += 1
        else:
            break
    return count

################################################################################
################################################################################
################################################################################

def make_one_x(L1,L2,ML,CP,CS):
    x_temp = [
            slope_n(L2,3),
            slope_n(L2,5),
            slope_n(L2,7),
            sign_present(L1,L2,2),
            sign_present(L1,L2,7),
            sign_present(L1,L2,15),
            last_n(L1,5),
            last_n(L2,5),
            last_n(L1,32),
            last_n(L2,32),
            np.average(CP),
            np.std(L1.iloc[-12:]),
            np.std(ML.iloc[-12:]),
            find_l1_peak_slope(L1,L2,CS),
            find_l1_peak_ratio(L1,L2,CS),
            find_drop(L1),
            find_drop(L2),
            cs_change(CS),
            get_kinks(L2),
            time_up(L2),
            time_down(L1)
        ]
    x_temp.extend(first_derivative(L2))  
    return x_temp

def make_x_300(L1,L2,ML,CP,CS,tag):
    x_temp = []
    y_temp = []
    n = 61
    k = 4
    for i in range(k):
        l1 = L1[i*n:(i+1)*n].reset_index().iloc[:,-1]
        l2 = L2[i*n:(i+1)*n].reset_index().iloc[:,-1]
        ml = ML[i*n:(i+1)*n].reset_index().iloc[:,-1]
        cs = CS[i*n:(i+1)*n].reset_index().iloc[:,-1]
        cp = CP[i*n:(i+1)*n].reset_index().iloc[:,-1]

        x_temp.append(make_one_x(l1,l2,ml,cp,cs))
        if i  == (k-1) and tag == 'True':
            y_temp.append(1)
        else:
            y_temp.append(0)
    return x_temp,y_temp

def make_x_3600(L1,L2,ML,CP,CS):
    x_temp = []
    y_temp = []
    index_temp = []
    n = 61
    k = 59
    for i in range(k):
        l1 = L1[i*n:(i+1)*n].reset_index().iloc[:,-1]
        l2 = L2[i*n:(i+1)*n].reset_index().iloc[:,-1]
        ml = ML[i*n:(i+1)*n].reset_index().iloc[:,-1]
        cs = CS[i*n:(i+1)*n].reset_index().iloc[:,-1]
        cp = CP[i*n:(i+1)*n].reset_index().iloc[:,-1]
        x_temp.append(make_one_x(l1,l2,ml,cp,cs))
        y_temp.append(0)
    return x_temp,y_temp

def make_cont_x(L1,L2,ML,CP,CS):
    x_temp = pd.DataFrame()
    cp = np.mean(CP)
    # this is the default value in case the iteration needs to be skipped
    default_x = [0,-9999,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(0,len(L1)-63):
        l1 = L1[i:i+61].reset_index().iloc[:,-1]
        l2 = L2[i:i+61].reset_index().iloc[:,-1]
        ml = ML[i:i+61].reset_index().iloc[:,-1]
        cs = CS[i:i+61].reset_index().iloc[:,-1]

        if ((cs.iloc[-1] < 0.61) or ( ((l2.iloc[-1]-l2.iloc[-4]) < 2) and ((l2.iloc[-1]-l2.iloc[-15]) < 2) and ((l2.iloc[-1]-l2.iloc[-32]) < 2))):
            x_temp = x_temp.append([default_x])
        else:
            x_temp = x_temp.append([make_one_x(l1,l2,ml,cp,cs)])
    return x_temp

################################################################################
################################################################################

def make_parallel_prob(L1,L2,ML,CS):
    prob_temp = []
    # this is the default value in case the iteration needs to be skipped
    default_prob = 0
    prob_temp = list(np.zeros(63))
    for i in range(0,len(L1)-63):
        l1 = L1[i:i+61].reset_index().iloc[:,-1]
        l2 = L2[i:i+61].reset_index().iloc[:,-1]
        ml = ML[i:i+61].reset_index().iloc[:,-1]
        cs = CS[i:i+61].reset_index().iloc[:,-1]
        # this is the check 2, the feature calculation is skipped if this check is not passed
        if ((cs.iloc[-1] < 0.45) or ( ((l2.iloc[-1]-l2.iloc[-4]) < 2) and ((l2.iloc[-1]-l2.iloc[-15]) < 2) and ((l2.iloc[-1]-l2.iloc[-32]) < 2))):
            prob_temp.append(default_prob)
        else:
            prob_temp.append(make_one_prob(l1,l2,ml,cs))
    return prob_temp

def get_cat(x_temp):
    x2 = x_temp[1]
    x7 = x_temp[5]
    x9 = x_temp[7]
    if x2 < 8.35:
        if x7 < 13.314:
            if x9 < 18.418:
                tt = -3.818367
            elif x9 >= 18.418:
                tt = 3.110193
        elif x7 >= 13.314:
            tt = 5.4519989
    elif x2 >= 8.35:
        if x9 < 7.7525:
            tt = 0.7469833
        elif x9 >= 7.7525:
            tt = 6.9635271
    return tt

def make_one_prob(L1,L2,ML,CS):

    x_temp = [
            slope_n(L2,3),
            slope_n(L2,5),
            slope_n(L2,7),
##            sign_present(L1,L2,2),
##            sign_present(L1,L2,7),
            sign_present(L1,L2,15),
            last_n(L1,5),
            last_n(L2,5),
##            last_n(L1,32),
            last_n(L2,32),
##            np.average(CP),
##            np.std(L1.iloc[-12:]),
##            np.std(ML.iloc[-12:]),
##            find_l1_peak_slope(L1,L2,CS),
##            find_l1_peak_ratio(L1,L2,CS),
            find_drop(L1),
##            find_drop(L2),
##            cs_change(CS),
##            get_kinks(L2),
            time_up(L2),
            time_down(L1)
        ]
    x_temp.append(first_derivative(L2)[0]) 

    coef = [-0.49276952,
            -0.20658922,
            0.73361965,
            -0.08565168,
            0.02515377,
            -0.11409218,
            0.28129691,
            0.19122434,
            -0.18042844,
            -0.11736072,
            0.62639931]
    inter = -6.36096808
    ret_tt = inter + np.inner(coef,x_temp)
    ret_tt = np.exp(ret_tt)/(1+np.exp(ret_tt))
    return ret_tt
