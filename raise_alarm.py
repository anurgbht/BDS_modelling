import pandas as pd
import numpy as np
import os
import implement_functions as mf
import time

def get_prob_series(col):
    count = 0
    output = []
    for i in range(len(col)):
        if col[i] > 0.85:
            count += 1
            if count > 3:
                output.append([col[i],i,count])
        else:
            count = 0
    return output

def get_prob_series_adjusted(col):
    count = 0
    output = []
    for i in range(len(col)):
        if col[i] > 0.9:
            count += 1
            if count > 5:
                output.append([col[i],i,count])
        else:
            count = 0
    return output

def get_actual_time(row):
    row = list(row)
    temp = row[-1]
    tt= []
    for i in temp:
        tt.append(i[-2]+63-i[-1])
    return list(np.unique(tt))

def get_bunch(col):
    count = 1
    tt = [count]
    for i in range(len(col)-1):
        if col[i+1] - col[i] > 20:
            count += 1
            tt.append(count)
        else:
            tt.append(count)
    return tt

########################################################################################################
########################################################################################################
########################################################################################################

os.chdir("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/prediction_test")
file_list = os.listdir()
# d5122018_predicted
# T9271014_086_1_3_10_predicted
##file_list = ['p7240319_predicted.csv']
total_no_files = len(file_list)
count_iter = 0
##f = open('sample_prediction.csv','w',buffering = 1)
t_string = ['L1-L2','L1-L2 adjusted','L2-L3','L2-L3 adjusted']
##t_string = ['L1-L2','L2-L3','L1-L2 adjusted','L2-L3 adjusted']
g_all = pd.DataFrame()
start_time = time.time()
for file_name in file_list:
    count_iter += 1
    print(file_name,count_iter,total_no_files)
    file = pd.read_csv(file_name)
    ## -1
##    col_no = (file.shape[1]-1)/4
    col_no = (file.shape[1])/4
    active_list = mf.give_active_list(col_no)
    tt = []
    for i in range(len(active_list)*4):
        i_temp = int(i/4)
        ## +1
##        temp_col = file.iloc[:,i+1]
        temp_col = file.iloc[:,i]
        
        t_rem = np.remainder(i,4)
        if ((t_rem == 0) or (t_rem == 2)):
            tt.append([file_name.split('_')[0],'TC_'+str(active_list[i_temp]),t_string[t_rem],get_prob_series(temp_col)])
        else:
            tt.append([file_name.split('_')[0],'TC_'+str(active_list[i_temp]),t_string[t_rem],get_prob_series_adjusted(temp_col)])
##tt = [print(x[0],x[-1][0][1]) for x in tt if len(x[-1]) > 0]
    tt = [x for x in tt if len(x[-1]) > 0]
    if len(tt) > 1:
        tt3 = []
        for i in tt:
            for j in i[-1]:
                tt3.append([i[0],i[1],i[2],j[-1],j[-2]-j[-1]+63])
        tt3 = pd.DataFrame(tt3)
        g1 = tt3.groupby(by=[0,1,2,4]).max().add_suffix('_Count').reset_index()
        g1.columns = ['file_name','TC_no','type','actual_time','length']
        g1 = g1.sort_values(['actual_time']).reset_index()
        g1.drop('index',1,inplace = True)
        t_bunch = pd.DataFrame(get_bunch(g1.loc[:,'actual_time']))
        g1 = pd.concat([g1,t_bunch],axis=1)
        g1.columns = ['file_name','TC_no','type','actual_time','length','bunch']
        g_all = pd.concat([g_all,g1],axis=0)


##print(g1)
print('Printing to - ' + os.getcwd())
g_all.to_csv('results.csv',index=False)
print('time taken per file = ',(time.time()-start_time)/len(file_list))
# we need two things, is the prob real ?
# can we see that the two probs are the same ?
# find the valid probs and then see which one of them triggers !
# but the problem is that both of these have to be simultaneous.

# write for normal and adjusted separately

##    for i in tt:
##        if len(i[-1])>0:
##            for j in i[-1]:
##                f.write(file_name.split('_')[0]+' , '+str(i[0])+' , '+str(i[1])+','+str(j).replace('[','').replace(']','')+'\n')
##                f.flush()
##f.close()
