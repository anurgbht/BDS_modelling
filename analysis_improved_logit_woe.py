import os
import pandas as pd
import numpy as np
import my_functions_no_keras as mf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
import time

def include_tree(row):
    x3 = row[2]
    x10 = row[9]
    x16 = row[15]
    x14 = row[13]
    x17 = row[16]
    x3 = row[2]

    if x3 <= 8.345:
        if x3 <= 5.25:
            tt = 'A'
        elif x3 > 5.25:
            if x14 <= -1.445:
                if x17 <= 0.045:
                    tt = 'B'
                elif x17 > 0.045:
                    tt = 'C'
            elif x14 > -1.445:
                tt = 'D'
    elif x3 > 8.345:
        if x16 <= 9.45:
            if x14 <= -0.405:
                tt = 'E'
            elif x14 > -0.405:
                tt = 'F'
        elif x16 > 9.45:
            tt = 'G'
        
    return tt
            

def subs_WOE(row):
    temp = woe_table.loc[woe_table.iloc[:,0] == row.iloc[-1],1]
    return temp

###########################################################################################
######################################~ MAIN ~#############################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/')

temp = pd.read_csv('data_dump_27_4_17.csv')
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
##beta_0 = clf_logit_trigger.intercept_
##beta = clf_logit_trigger.coef_
##
##clf_RF_trigger = mf.prelim_RF(data_logit,Y_all_clean,0.1)
##clf_gbm_trigger = mf.prelim_gbm(data_logit,Y_all_clean,0.1)

##clf_keras_trigger = mf.prelim_keras(data_logit,Y_all_clean,0.1)

###########################################################################################
###################################Continuous Analysis######################################
###########################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/test/')
file_list = os.listdir()
##file_list= ["F1031504_097_01_3_14_CASTER-B_True_JAN'15_new.csv"]
##random.shuffle(file_list)
count = 0
duration_all = 0
for file_name in file_list:
    count = count + 1
    print(count,file_name)
    if count < 5000:
        file = pd.read_csv(file_name)
        TC_layer = int(np.mean(file.loc[:,'TC_layer']))
        
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)
        tt3 = 'TC' + str(TC_layer + 40)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        L3 = file.loc[:,tt3]
        
        ML = file.loc[:,'M.level']
        CS = file.loc[:,'C.speed']
        CP = file.loc[:,'C.percent']
        MW = file.loc[:,'M.width']

        TC_layer_left = mf.find_left(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_left)
        tt2 = 'TC' + str(TC_layer_left + 20)
        tt3 = 'TC' + str(TC_layer_left + 40)

        LL1 = file.loc[:,tt1]
        LL2 = file.loc[:,tt2]
        LL3 = file.loc[:,tt3]

        TC_layer_right = mf.find_right(MW[0],TC_layer)

        tt1 = 'TC' + str(TC_layer_right)
        tt2 = 'TC' + str(TC_layer_right + 20)
        tt3 = 'TC' + str(TC_layer_right + 40)

        LR1 = file.loc[:,tt1]
        LR2 = file.loc[:,tt2]
        LR3 = file.loc[:,tt3]

        TC_layer_opp = mf.find_opposite(TC_layer)

        tt1 = 'TC' + str(TC_layer_opp)
        tt2 = 'TC' + str(TC_layer_opp + 20)
        tt3 = 'TC' + str(TC_layer_opp + 40)
        LO1 = file.loc[:,tt1]
        LO2 = file.loc[:,tt2]
        LO3 = file.loc[:,tt3]

        start_time = time.time()
        print("Calculating for layer 2-1")
        # making the continuous x's
        temp_x_trigger = mf.make_cont_x(L1,L2,ML,CP,CS,MW)
        # print(temp_x_trigger.iloc[0,10]," ye hai vo")
        # including the tree based information
        temp = temp_x_trigger.apply(include_tree,axis=1)
        temp_data = pd.concat([temp_x_trigger,temp],axis=1)
        temp2 = temp_data.apply(subs_WOE,axis=1)
        temp_x_trigger = pd.concat([temp_x_trigger,temp2],axis=1)

        # making the continuous x's
        temp_x_left = mf.make_cont_x(LL1,LL2,ML,CP,CS,MW)
        # including the tree based information
        temp = temp_x_left.apply(include_tree,axis=1)
        temp_data = pd.concat([temp_x_left,temp],axis=1)
        temp2 = temp_data.apply(subs_WOE,axis=1)
        temp_x_left = pd.concat([temp_x_left,temp2],axis=1)

        # making the continuous x's
        temp_x_right = mf.make_cont_x(LR1,LR2,ML,CP,CS,MW)
        # including the tree based information
        temp = temp_x_right.apply(include_tree,axis=1)
        temp_data = pd.concat([temp_x_right,temp],axis=1)
        temp2 = temp_data.apply(subs_WOE,axis=1)
        temp_x_right = pd.concat([temp_x_right,temp2],axis=1)
        
        logit_trigger = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger))
        logit_left = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_left))
        logit_right = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_right))

        # for layer 3 now
        print("Calculating for layer 3-2 now")
        
        # making the continuous x's
        temp_x_trigger3 = mf.make_cont_x(L2,L3,ML,CP,CS,MW)
        # including the tree based information
        temp3 = temp_x_trigger3.apply(include_tree,axis=1)
        temp_data3 = pd.concat([temp_x_trigger3,temp3],axis=1)
        temp23 = temp_data3.apply(subs_WOE,axis=1)
        temp_x_trigger3 = pd.concat([temp_x_trigger3,temp23],axis=1)

        # making the continuous x's
        temp_x_left3 = mf.make_cont_x(LL2,LL3,ML,CP,CS,MW)
        # including the tree based information
        temp3 = temp_x_left3.apply(include_tree,axis=1)
        temp_data3 = pd.concat([temp_x_left3,temp3],axis=1)
        temp23 = temp_data3.apply(subs_WOE,axis=1)
        temp_x_left3 = pd.concat([temp_x_left3,temp23],axis=1)

        # making the continuous x's
        temp_x_right3 = mf.make_cont_x(LR2,LR3,ML,CP,CS,MW)
        # including the tree based information
        temp3 = temp_x_right3.apply(include_tree,axis=1)
        temp_data3 = pd.concat([temp_x_right3,temp3],axis=1)
        temp23 = temp_data3.apply(subs_WOE,axis=1)
        temp_x_right3 = pd.concat([temp_x_right3,temp23],axis=1)
        
        logit_trigger3 = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_trigger3))
        logit_left3 = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_left3))
        logit_right3 = pd.DataFrame(clf_logit_trigger.predict_proba(temp_x_right3))

        end_time = time.time()

        plt.clf()
        plt.figure(num=None, figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')

        plt.subplot(3,3,1)
        plt.plot(range(len(logit_trigger.iloc[:,-1])),logit_trigger.iloc[:,-1],range(len(logit_trigger3.iloc[:,-1])),logit_trigger3.iloc[:,-1])
        plt.xlabel('Time')
        plt.ylabel('Probability of being a sticker')
        plt.legend(["layer 2","layer 3"])
        plt.ylim([0,1])
        plt.title('Trigger')
        plt.grid(1)

        plt.subplot(3,3,4)
        plt.plot(range(len(L2)),L2,range(len(L2)),L1,range(len(L2)),L3)
        plt.legend(['layer 2','layer 1','layer 3'])
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.grid(1)

        plt.subplot(3,3,7)
        plt.plot(range(len(ML)),ML,range(len(CS)),100*CS)
        plt.xlabel('Time')
        plt.legend(['Mold Level','100x Casting Speed'])
        plt.grid(1)

        plt.subplot(3,3,2)
        plt.plot(range(len(logit_left.iloc[:,-1])),logit_left.iloc[:,-1],range(len(logit_left3.iloc[:,-1])),logit_left3.iloc[:,-1])
        plt.xlabel('Time')
        plt.ylabel('Probability of being a sticker')
        plt.legend(["layer 2","layer 3"])
        plt.ylim([0,1])
        plt.title('Left')
        plt.grid(1)

        plt.subplot(3,3,5)
        plt.plot(range(len(LL2)),LL2,range(len(LL2)),LL1,range(len(LL2)),LL3)
        plt.legend(['layer 2','layer 1','layer 3'])
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.grid(1)

        plt.subplot(3,3,8)
        plt.plot(range(len(ML)),ML,range(len(CS)),100*CS)
        plt.xlabel('Time')
        plt.legend(['Mold Level','100x Casting Speed'])
        plt.grid(1)

        plt.subplot(3,3,3)
        plt.plot(range(len(logit_right.iloc[:,-1])),logit_right.iloc[:,-1],range(len(logit_right3.iloc[:,-1])),logit_right3.iloc[:,-1])
        plt.xlabel('Time')
        plt.ylabel('Probability of being a sticker')
        plt.legend(["layer 2","layer 3"])
        plt.ylim([0,1])
        plt.title('Right')
        plt.grid(1)

        plt.subplot(3,3,6)
        plt.plot(range(len(LR2)),LR2,range(len(LR2)),LR1,range(len(LR2)),LR3)
        plt.legend(['layer 2','layer 1','layer 3'])
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.grid(1)

        plt.subplot(3,3,9)
        plt.plot(range(len(ML)),ML,range(len(CS)),100*CS)
        plt.xlabel('Time')
        plt.legend(['Mold Level','100x Casting Speed'])
        plt.grid(1)

        plt.suptitle(file_name.split('.')[0] )
        plot_save = 'D://Confidential//Projects//Steel//LD2 BDS//prelim_analysis//data//pickle_test//' + file_name.split('.')[0] + '_pred.png'
        plt.savefig(plot_save)
        plt.close()

        duration_temp = end_time-start_time
        duration_all = duration_all + duration_temp
        print("time for this iteration, for all ",duration_temp,duration_all)
