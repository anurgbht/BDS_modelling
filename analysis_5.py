import os
import numpy as np
import pandas as pd
import implement_functions as mf

########################################################################################
#################################~MAIN~################################################
########################################################################################

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/training_files')
all_files = os.listdir()
ttt=(len(all_files))
temp=pd.DataFrame()
count = 0
X_all_trigger = pd.DataFrame()
Y_all = pd.DataFrame()
index_file = pd.DataFrame()
for file_name in all_files:
    count = count + 1
    print(count,"/",ttt)
    file = pd.read_csv(file_name)
    if int(file_name.split('_')[3]) == 2 :
        TC_layer = int(file_name.split('_')[4])+1
        tt1 = 'TC' + str(TC_layer)
        tt2 = 'TC' + str(TC_layer + 20)

        L1 = file.loc[:,tt1]
        L2 = file.loc[:,tt2]
        ML = file.loc[:,'M.level']
        CS = file.loc[:,'C.speed']
        CP = file.loc[:,'C.percent']
        MW = file.loc[:,'M.width']

        tx,ty = mf.make_x(L1,L2,ML,CP,CS,file_name.split('_')[6])
        X_all_trigger = X_all_trigger.append(tx)

        Y_all = Y_all.append(ty)

temp = pd.concat([X_all_trigger,Y_all,index_file],axis=1)
print(temp.shape)
temp = temp.dropna()
print(temp.shape)

temp.to_csv('data_dump_21_6_17.csv',index=False)


##X_all_left = pd.DataFrame(pp.scale(temp.iloc[:,0:17]))
##X_all_trigger = pd.DataFrame(pp.scale(temp.iloc[:,17:34]))
##X_all_right = pd.DataFrame(pp.scale(temp.iloc[:,32:48]))
##Y_all_clean = temp.iloc[:,48]
##index_file_clean = temp.iloc[:,49]
##
##print(X_all_trigger.shape,' : Size of all X trigger')
##print(X_all_left.shape,' : Size of all X left')
##print(X_all_right.shape,' : Size of all X right')
##print(Y_all_clean.shape,' : Size of all Y')
##print(index_file_clean.shape,' : Shape of index file')



##clf_RF = prelim_RF(X_all,Y_all)
##clf_logit = prelim_logit(X_all,Y_all)
##clf_GBM_trigger,y_test,my_pred,test_index = prelim_GBM(X_all_trigger,Y_all_clean,0.2)
##clf_GBM_left,y_test,my_pred,test_index = prelim_GBM(X_all_left,Y_all_clean,0.2)
##clf_GBM_right,y_test,my_pred,test_index = prelim_GBM(X_all_right,Y_all_clean,0.2)
##
##my_pred_trigger = my_pred_fun(clf_GBM_trigger,X_all_trigger,Y_all_clean,0.2)
##print(confusion_matrix(Y_all_clean,my_pred_trigger))
##my_pred_left = my_pred_fun(clf_GBM_left,X_all_left,Y_all_clean,0.2)
##print(confusion_matrix(Y_all_clean,my_pred_left))
##my_pred_right = my_pred_fun(clf_GBM_right,X_all_right,Y_all_clean,0.2)
##print(confusion_matrix(Y_all_clean,my_pred_right))
##
##my_pred_trigger = clf_GBM_trigger.predict_proba(X_all_trigger)
##my_pred_left = clf_GBM_left.predict_proba(X_all_left)
##my_pred_right = clf_GBM_right.predict_proba(X_all_right)

##my_pred_trigger.to_csv('my_pred_trigger.csv')
##my_pred_left.to_csv('my_pred_left.csv')
##my_pred_right.to_csv('my_pred_right.csv')
##Y_all_clean.to_csv('Y_all.csv')

##index_false = pd.DataFrame()
##
##for i in range(len(my_pred)):
##    if (Y_all_clean.iloc[i][0] != my_pred.iloc[i]) & Y_all_clean.iloc[i][0] == 1:
##        index_false = index_false.append([i])
##
##misclassified_files = [index_file.iloc[x][0] for x in index_false.iloc[:,0]]
##print(misclassified_files)
