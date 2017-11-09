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
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout


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

def prelim_keras(*args):
    print('Preliminary Keras Analysis')
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
        input_dim  = X_train.shape[1]
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = Sequential()
        clf.add(Dense(15, input_dim=input_dim, init='uniform', activation='relu'))
##        clf.add(Dropout(0.3))
        clf.add(Dense(5, activation='sigmoid'))
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

def prelim_RF(*args):
    print('Preliminary Random Forest Analysis')
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
    if len(args) == 5:
        x_data = args[0]
        y_data = args[1]
        file_names = args[2]
        thresh = args[3]
        plot_flag = args[4]
        X_train, X_test, y_train, y_test,test_index = train_test_split_my(x_data, y_data, test_size=0.3)
        proceed_flag = 1
    elif len(args) == 7:
        X_train = args[0]
        X_test = args[1]
        y_train = args[2]
        y_test = args[3]
        file_names = args[4]
        thresh = args[5]
        plot_flag = args[6]
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
        clf = LogisticRegression( max_iter=10000)
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
        temp1 = list(my_pred <= thresh)
        temp2 = list(y_test.iloc[:,0] == 1)
        temp = [x and y for (x,y) in zip(temp1,temp2)]
        tt_file = file_names.loc[temp]
        print(confusion_matrix(y_test,my_pred))
        out_files = pd.concat([file_names,y_test,pd.DataFrame(clf.predict_proba(X_test))],axis=1)
        print(accuracy_score(y_test,my_pred),' : My accuracy score')
        return clf,tt_file,out_files
    else:
        print('Something went wrong')

def prelim_svm(*args):
    print('Preliminary SVM Analysis')
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
        clf = SVC(probability=True,kernel='rbf', C=.5,gamma=1)
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

def find_left_list(therm_list,TC_layer):
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
    
def find_right_list(therm_list,TC_layer):
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

def find_index(l2):
    n = len(l2)
    tt = -1
    for i in range(1,n-3):
        del1 = l2[-(i+0)] - l2[-(i+1)]
        del2 = l2[-(i+1)] - l2[-(i+2)]
        if ((del1<=0)and(del2>0)):
            tt=n-(i)
            break
    return tt

def find_index_l1(l1,l2_index):
    l1 = l1[:l2_index]
    tt = find_index(l1)
    if tt < 0:
        tt = l1.index(max(l1))
    return tt-1

################################################################################
###########################FEATURE GENERATION HERE##############################
################################################################################

def rise_slope(l2,max_index):
    count = 0
    rise = 0
    k=2
    for i in range(max_index-k):
        if l2[max_index-i]>l2[max_index-(i+k)]:
            count += 1
            rise  = (l2[max_index] -l2[max_index-(i+k)])/(count+1)
        else:
            break
    return rise

def crossover(l1,l2,max_index):
    if max_index-20 > 5:
        far = np.average(l2[:max_index-20]) - np.average(l1[:max_index-20])
    else:
        far = np.average(l2[:5]) - np.average(l1[:5])
    near = l2[max_index] - l1[max_index]
    tt = (near-far)/(abs(far)+1)
    return tt

def crossover_extended(l1,l2,max_index):
    if max_index-20 > 5:
        far = np.average(l2[:max_index-20]) - np.average(l1[:max_index-20])
    else:
        far = np.average(l2[:5]) - np.average(l1[:5])
    near = l2[max_index] - l1[max_index]
    tt = (near-far)/(abs(far)+1)
    return [tt,near,far]
    
def rise_amp(l2,max_index):
    rise = 0
    k=2
    if max_index > 1:
        for i in range(max_index-k):
            if l2[max_index-i]>l2[max_index-(i+k)]:
                rise  = (l2[max_index] -l2[max_index-(i+k)])
                check_index = max_index - i
            else:
                rise  = (l2[max_index] -l2[max_index-(i+1)])
                check_index = max_index - (i+1)
                break
        if check_index >=10:
            rise = rise - (max(l2[check_index-10:check_index+1]) - l2[check_index])
        else:
            rise = rise - (max(l2[:check_index+1]) - l2[check_index])
    else:
        rise = 0.000001
##    print(rise,check_index,max_index,l2)
    return rise

def drop_slope(l2,l1_index,l2_index):
    drop = (l2[l1_index]-l2[l2_index])/(l2_index-l1_index)
    return drop

def drop_amp(l2,l1_index,l2_index):
    drop = (l2[l1_index]-l2[l2_index])
    return drop
    

def rise_time(l2,max_index):
    count = 0
    k = 2
    for i in range(0,max_index-k):
            if l2[max_index-i]>l2[max_index-(i+k)]:
                count += 1
            else:
                break
    return count

def drop_time(l2,max_index):
    count = 0
    k = 2
    for i in range(max_index,len(l2)-k):
        if l2[i]>l2[i+k]:
            count += 1
        else:
            break
    return count

def time_delta(l1,l2,cs,max_index1,max_index2):
    if max_index1<max_index2:
        tt = np.average(cs[max_index1:max_index2])
    else:
        tt = 0
    return tt*(max_index2 - max_index1)
    
################################################################################
################################################################################
################################################################################

def make_one_x(L1,L2,CS,l1_index,l2_index,file_name):
    x_temp = [
            file_name,
            rise_slope(L1,l1_index),
            rise_slope(L2,l2_index),
            crossover(L1,L2,l2_index),
            rise_amp(L1,l1_index),
            rise_amp(L2,l2_index),
            drop_slope(L1,l1_index,l2_index),
            drop_amp(L1,l1_index,l2_index),
            rise_time(L2,l2_index),
            drop_time(L1,l1_index),
            time_delta(L1,L2,CS,l1_index,l2_index)
        ]
    return x_temp

def make_x_300_stay(L1,L2,CS,file_name):
    L1 = list(L1)
    L2 = list(L2)
    CS = list(CS)
    temp = L2[230:255]
    max_index = temp.index(max(temp))
    # as compared to max, calculate the features
    x_temp = []
    y_temp = []
    iter_index = 230+max_index+1
    l1 = L1[iter_index-60:iter_index]
    l2 = L2[iter_index-60:iter_index]
    cs = CS[iter_index-60:iter_index]
    # find_index can be used here as well
    l2_index = 59
    l1_index = find_index_l1(l1,l2_index+1)
##    print(l1)
##    print(l2)
##    print(l1_index,l2_index)
##    print(l1[l1_index],l2[l2_index])
    q = make_one_x(l1,l2,cs,l1_index,l2_index,file_name)
    x_temp.append(q)
    y_temp.append(1)
    return x_temp,y_temp

##def make_x_300(L1,L2,CS,file_name):
##    L1 = list(L1)
##    L2 = list(L2)
##    CS = list(CS)
##    temp = L2[230:250]
##    max_index = temp.index(max(temp))
##    iter_index = 230+max_index+3
##    x_temp = []
##    y_temp = []
##    l1 = L1[iter_index-60:iter_index]
##    l2 = L2[iter_index-60:iter_index]
##    cs = CS[iter_index-60:iter_index]
##    # find_index can be used here as well
##    l2_index = 57
##    l1_index = find_index_l1(l1,l2_index)
##    temp = l2
##    del1 = temp[-1] - temp[-2]
##    del2 = temp[-2] - temp[-3]
##    del3 = temp[-3] - temp[-4]
##    del4 = temp[-4] - temp[-5]
##    del5 = temp[-5] - temp[-6]
##    if ((del1<0)and(del2<0)and(del3>0)and(del4>0)):
##        x_temp.append(make_one_x(l1,l2,cs,l1_index,l2_index,file_name))
##        y_temp.append(1)
##    return x_temp,y_temp

def make_x_online(L1,L2,CS,TC_layer,file_name):
    L1 = list(L1)
    L2 = list(L2)
    CS = list(CS)
    x_temp = []
    y_temp = []
    for i in range(10,len(L2)):
##        print(i)
        temp = L2[i-10:i]
        del1 = temp[-1] - temp[-2]
        del2 = temp[-2] - temp[-3]
        del3 = temp[-3] - temp[-4]
        del4 = temp[-4] - temp[-5]
        del5 = temp[-5] - temp[-6]
        if ((del1<=0)and(del2>0)and(del3>0)):
            # do i<60 here for larger files
            if i < 60:
                l1 = L1[:i]
                l2 = L2[:i]
                cs = CS[:i]
                l2_index = i-1
            else:
                l1 = L1[i-60:i]
                l2 = L2[i-60:i]
                cs = CS[i-60:i]
                l2_index = 58
            if min(cs) > 0.6:
                l1_index = find_index_l1(l1,l2_index)
                tt = make_one_x(l1,l2,cs,l1_index,l2_index,file_name)
                q = make_one_x(l1,l2,cs,l1_index,l2_index,file_name)
                x_temp.append(q)
                y_temp.append(0)
    return x_temp,y_temp

################################################################################
################### SECOND LEVEL FEATURE GENERATION ###########################
################################################################################

def prob_level_two(temp_file,active_list,tag):
    tt = []
    ml = list(temp_file.loc[:,'M.level'])
    cs = list(temp_file.loc[:,'C.speed'])
    for TC_layer in active_list:
        if tag == '12':
            l1 = list(temp_file.loc[:,'TC' + str(TC_layer)])
            l2 = list(temp_file.loc[:,'TC' + str(TC_layer+20)])
        elif tag == '23':
            l1 = list(temp_file.loc[:,'TC' + str(TC_layer+20)])
            l2 = list(temp_file.loc[:,'TC' + str(TC_layer+40)])
        del1 = l2[-1] - l2[-2]
        del2 = l2[-2] - l2[-3]
        del3 = l2[-3] - l2[-4]
        del4 = l2[-4] - l2[-5]
        if (((del1<=0)and(del2>0)and(del3>0)and(del4>0))and(cs[-1]>0.6)):
            l2_index = len(l2)-2
            l1_index = find_index_l1(l1,l2_index)
            tt.append(make_one_prob_level_two(l1,l2,ml,cs,l1_index,l2_index))
        else:
            tt.append(-9)
    return tt

def make_one_prob_level_two(L1,L2,ML,CS,l1_index,l2_index):

    x_temp = [
            rise_slope(L1,l1_index),
            rise_slope(L2,l2_index),
            crossover(L1,L2,l2_index),
            rise_amp(L1,l1_index),
            rise_amp(L2,l2_index),
            drop_slope(L1,l1_index,l2_index),
            drop_amp(L1,l1_index,l2_index),
            rise_time(L2,l2_index),
            drop_time(L1,l1_index),
            time_delta(L1,L2,CS,l1_index,l2_index)
        ]
    coef =  [ 0.48022494,  0.1992056 , 0.1,  0.00936032,  0.6899287 ,
              0.65854065, -0.06402542, 0.08726375,  0.06120159, -0.02747763]
##    coef =  [ 0.48022494,  0.1992056 , -0.03366528,  0.00936032,  0.5899287 ,
##              0.85854065, -0.06402542, -0.18726375,  0.06120159, -0.02747763]
    inter = -6.5
    ret_tt = inter + np.inner(coef,x_temp)
    ret_tt = np.exp(ret_tt)/(1+np.exp(ret_tt))
    return ret_tt
        
##def all_features(temp_file,active_list):
##    tt = []
##    ml = list(temp_file.loc[:,'M.level'])
##    cs = list(temp_file.loc[:,'C.speed'])
##    for TC_layer in active_list:
##        l1 = list(temp_file.loc[:,'TC' + str(TC_layer)])
##        l2 = list(temp_file.loc[:,'TC' + str(TC_layer+20)])
##        del1 = l2[-1] - l2[-2]
##        del2 = l2[-2] - l2[-3]
##        del3 = l2[-3] - l2[-4]
##        if (((del1<=0)and(del2>0)and(del3>0))and(cs[-7]>0.6)):
##            l2_index = len(l2)-2
##            l1_index = find_index_l1(l1,l2_index)
##            print(l1_index,l2_index)
##            tt.append(make_one_feature(l1,l2,ml,cs,l1_index,l2_index))
##        else:
##            pass
##    return tt

##def make_one_feature(L1,L2,ML,CS,l1_index,l2_index):
##
##    x_temp = [
##            rise_slope(L1,l1_index),
##            rise_slope(L2,l2_index),
##            crossover(L1,L2,l2_index),
##            rise_amp(L1,l1_index),
##            rise_amp(L2,l2_index),
##            drop_slope(L1,l1_index,l2_index),
##            drop_amp(L1,l1_index,l2_index),
##            rise_time(L2,l2_index),
##            drop_time(L1,l1_index),
##            time_delta(L1,L2,CS,l1_index,l2_index)
##        ]
####    temp = get_cat(x_temp)
####    x_temp.append(temp)
##    coef =   [ 0.48022494,  0.1992056 , -0.03366528,  0.00936032,  0.5899287 ,
##         0.85854065, -0.06402542, -0.18726375,  0.06120159, -0.02747763]
##    inter = -6.5
##    ret_tt = inter + np.inner(coef,x_temp)
##    ret_tt = np.exp(ret_tt)/(1+np.exp(ret_tt))
##    x_temp.extend([l1_index,l2_index])
##    x_temp.append(ret_tt)
##    return x_temp

################################################################################
#################### THIRD LEVEL FEATURE GENERATION ############################
################################################################################

def prob_level_three(temp_file,active_list,i,file_name):
    tt = []
    ml = list(temp_file.loc[:,'M.level'])
    cs = list(temp_file.loc[:,'C.speed'])
    for TC_layer in active_list:
        l1 = list(temp_file.loc[:,'TC' + str(TC_layer)])
        l2 = list(temp_file.loc[:,'TC' + str(TC_layer+20)])
        del1 = l2[-1] - l2[-2]
        del2 = l2[-2] - l2[-3]
        del3 = l2[-3] - l2[-4]
        del4 = l2[-4] - l2[-5]
        if (((del1<=0)and(del2>0)and(del3>0)and(del4>0))and(cs[-1]>0.6)):
            l2_index = len(l2)-2
            l1_index = find_index_l1(l1,l2_index)
            q = make_one_prob_level_three(l1,l2,ml,cs,l1_index,l2_index)
            q.extend([i,file_name,TC_layer])
            tt.append(q)
        else:
            q = [-1, -1, -1, -1, -1, -1, -1,-1,i,file_name,TC_layer]
            tt.append(q)
    return tt
        
def make_one_prob_level_three(L1,L2,ML,CS,l1_index,l2_index):

    x_temp = [
            rise_slope(L1,l1_index),
            rise_slope(L2,l2_index),
            crossover(L1,L2,l2_index),
            rise_amp(L1,l1_index),
            rise_amp(L2,l2_index),
            drop_slope(L1,l1_index,l2_index),
            drop_amp(L1,l1_index,l2_index),
            rise_time(L2,l2_index),
            drop_time(L1,l1_index),
            time_delta(L1,L2,CS,l1_index,l2_index)
        ]
    coef =  [ 0.48022494,  0.1992056 , 0.1,  0.00936032,  0.5899287 ,
         0.85854065, -0.06402542, -0.18726375,  0.06120159, -0.02747763]
    inter = -6.5
    ret_tt = inter + np.inner(coef,x_temp)
    ret_tt = np.exp(ret_tt)/(1+np.exp(ret_tt))

    third_level_features = [
            long_term_layer_drop(L1,l2_index,1),
            long_term_layer_drop(L2,l2_index,2),
            l_avg(L2,l2_index),
            casting_speed_slope(CS,l2_index),
            mold_level_drop(ML,l2_index),
            high_freq(L2,l2_index),
            rise_time(L1,l1_index),
            drop_time(L2,l2_index)
        ]
    third_level_features = list(std_compare(L2,l2_index)) + third_level_features
    x_temp.extend(third_level_features)
    x_temp.extend(crossover_extended(L1,L2,l2_index))
    x_temp.append(ret_tt)

    return x_temp

################################################################################
################################################################################
################################################################################

def long_term_layer_drop(l2,l2_index,layer_tag):
    if layer_tag == 2:
        tt = l2[:l2_index-5]
        ret_tt = max(tt)-min(tt)
    else:
        ret_tt = max(l2)-min(l2)
    return ret_tt

def l_avg(l2,l2_index):
    tt = l2[:l2_index-5]
    ret_tt = np.average(tt)
    return ret_tt

def casting_speed_slope(cs,l2_index):
    tt = cs[:l2_index]
    ret_tt = max(tt)-min(tt)
    return ret_tt

def mold_level_drop(ml,l2_index):
    tt = ml[:l2_index]
    ret_tt = max(tt)-min(tt)
    return ret_tt

def std_compare(l2,max_index):
    k=2
    if max_index > 1:
        for i in range(max_index-k):
            if l2[max_index-i]>l2[max_index-(i+k)]:
                check_index = max_index - i
            else:
                check_index = max_index - (i+1)
                break
        std1 = np.std(l2[check_index:max_index+1])
        if check_index-30 >=0:
            std2 = np.std(l2[check_index-30:check_index+1])
        else:
            std2 = np.std(l2[:check_index+1])
    else:
        std1 = 0.000001
        std2 = 0.000001
    return std1,std2,check_index,max_index

def high_freq(l2,l2_index):
    temp = l2[:l2_index]
    temp2 = []
    for i in range(len(temp)-1):
        temp2.append(temp[i+1] - temp[i])
    temp4 = [1 if x>0 else -1 for x in np.sign(temp2)]
    count = 0
    for i in range(len(temp4)-1):
        if (temp4[i+1] != temp4[i]):
            count += 1
    return count

################################################################################
#################### AUXILIARY SUPPORTING FUNCTIONS ###########################
################################################################################

def get_cat(x_temp):
    x2 = x_temp[2]
    x4 = x_temp[4]
    if x4 < 12.45:
        tt = -5.31188811
    elif x4 >= 12.45:
        if x2 < 5.445:
            tt = -0.6965857
        elif x2 >= 5.445:
            tt = 5.8407746
    return tt

def transform_features(q,avg_l,constants):
    tt = [q[0]]
    for i in range(1,len(q)):
        m = constants[i-1][0]
        c = constants[i-1][1]
        if avg_l - 146 < 0:
            temp = q[i] - (m*(avg_l - 146) + c)
        else:
            temp = q[i]
        tt.append(temp)
    return tt

