import os
import pandas as pd
import numpy as np
import my_functions as mf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
import time

def include_tree(row):
    x2 = row[1]
    x6 = row[5]
    x7 = row[6]
    x8 = row[7]
    x13 = row[12]

    if x2 <= 6.85:
        if x7 <= 0.9846:
            if x6 <= 1.0045:
                tt = 'A'
            elif x6 > 1.0045:
                tt = 'B'
        elif x7 > 0.9846:
            tt = 'C'
    elif x2 > 6.85:
        if x13 <= 9.45:
            tt = 'D'
        elif x13 > 9.45:
            tt = 'E'
    return tt            

def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp

###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_24_3_17.csv')
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

X_all_left = pd.DataFrame(temp.iloc[:,0:20])
X_all_trigger = pd.DataFrame(temp.iloc[:,20:40])
X_all_right = pd.DataFrame(temp.iloc[:,40:60])
Y_all_clean = temp.iloc[:,60]
index_file_clean = temp.iloc[:,61]

X_all_trigger.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20"]
X_all_left.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20"]
X_all_right.columns = ["X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20"]

print(X_all_trigger.shape,' : Size of all X trigger')
print(X_all_left.shape,' : Size of all X left')
print(X_all_right.shape,' : Size of all X right')
print(Y_all_clean.shape,' : Size of all Y')
print(index_file_clean.shape,' : Shape of index file')

##from sklearn import tree
##import pydotplus
##clf = tree.DecisionTreeClassifier(min_samples_leaf=25,max_depth = 7)
##clf = clf.fit(X_all_trigger,Y_all_clean)
##dot_data = tree.export_graphviz(clf, out_file=None)
##graph = pydotplus.graph_from_dot_data(dot_data)
##pdf_name = "temp_vis_27_3_17" + ".pdf"
##graph.write_pdf(pdf_name)

temp = X_all_trigger.apply(include_tree,axis=1)

data_logit = pd.concat([X_all_trigger,temp],axis=1)

print(X_all_trigger.shape)
print(data_logit.shape)

global woe_table
woe_table = mf.calc_WOE(data_logit.iloc[:,-1],Y_all_clean)

temp2 = data_logit.apply(subs_WOE,axis=1)

# just to make sure everything is going okay
temp3 = pd.concat([temp2,data_logit.iloc[:,-1]],axis=1)

data_logit = pd.concat([X_all_trigger,temp2],axis=1)

clf_logit_trigger = LogisticRegression()
clf_logit_trigger =clf_logit_trigger.fit(data_logit,Y_all_clean)
beta_0 = clf_logit_trigger.intercept_
beta = clf_logit_trigger.coef_
##
##clf_RF_trigger = mf.prelim_RF(data_logit,Y_all_clean,0.1)
##clf_gbm_trigger = mf.prelim_gbm(data_logit,Y_all_clean,0.1)

##clf_keras_trigger = mf.prelim_keras(data_logit,Y_all_clean,0.1)

###########################################################################################
###################################Continuous Analysis######################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/layer 3 files')
file_list = os.listdir()
##file_list= ["H2020210_110_01_2_17_true.csv"]
##random.shuffle(file_list)
for file_name in file_list:
    print(file_name)
    file = pd.read_csv(file_name)
##    file = file.iloc[2600:2800,:].reset_index()
    ML = file.loc[:,'M.level']
    CS = file.loc[:,'C.speed']
    CP = file.loc[:,'C.percent']
    MW = file.loc[:,'M.width']
    plt.clf()
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    active_range = mf.find_active(MW[0])
    count = 0
    for TC_layer in active_range:
        count += 1
        print(count,len(active_range))
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)
        tt3 = 'TC' + str(TC_layer + 40)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        L3 = file.loc[:,tt3]

        plt.subplot(7,3,count)
        plt.plot(range(len(L1)),L1,range(len(L1)),L2,range(len(L1)),L3)
        plt.legend(['Layer 1','Layer 2'])
        plt.ylabel('TC ' + str(TC_layer))
        plt.grid(1)


    plt.subplot(7,3,count + 1)
    plt.plot(range(len(ML)),ML,range(len(CS)),100*CS)
    plt.legend(['ML','100x CS'])
    plt.yticks([60])
    plt.grid(1)

    plt.suptitle(file_name.split('.')[0]+" casting speed = "+ str(np.mean(CS)))
    plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//layer_3//' + file_name.split('.')[0] + '_TC_all.png'
    plt.savefig(plot_save)
    plt.close()

for file_name in file_list:
    print(file_name)
    file = pd.read_csv(file_name)
##    file = file.iloc[2600:2800,:].reset_index()
    ML = file.loc[:,'M.level']
    CS = file.loc[:,'C.speed']
    CP = file.loc[:,'C.percent']
    MW = file.loc[:,'M.width']
    plt.clf()
    plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    active_range = mf.find_active(MW[0])
    count = 0
    for TC_layer in active_range:
        count += 1
        print(count,len(active_range))
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]

        TC_layer_opp = mf.find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]

        # making the continuous x's
        temp_x_trigger = mf.make_cont_x(L1,L2,ML,CP,CS,LO1,LO2,MW)
        # including the tree based information
        temp = temp_x_trigger.apply(include_tree,axis=1)
        temp_data = pd.concat([temp_x_trigger,temp],axis=1)
        temp2 = temp_data.apply(subs_WOE,axis=1)
        temp_x_trigger = pd.concat([temp_x_trigger,temp2],axis=1)
        
        logit_trigger = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger))

        plt.subplot(7,3,count)
        plt.plot(range(len(logit_trigger.iloc[:,-1])),logit_trigger.iloc[:,-1])
##        plt.xlabel('Time')
        plt.ylabel('TC ' + str(TC_layer))
        plt.ylim([0,1])
        plt.grid(1)


    plt.subplot(7,3,count + 1)
    plt.plot(range(len(ML)),ML,range(len(CS)),100*CS)
    plt.legend(['ML','100x CS'])
    plt.grid(1)

    plt.suptitle(file_name.split('.')[0] )
    plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//plots//TN_pred//dat_24_4//' + file_name.split('.')[0] + '_pred_all.png'
    plt.savefig(plot_save)
    plt.close()
