# importing the necessary modules
import multiprocessing as mp
import time
import numpy as np
import random
import os
import pandas as pd
import pickle

# this function incorporates a tree based classifier in the modelling
# just a part of feature generation process

def include_tree(row):
    x3 = row[2]
    x10 = row[9]
    x16 = row[15]
    x14 = row[13]
    x17 = row[16]
    x3 = row[2]

    if x3 <= 8.345:
        if x3 <= 5.25:
            ## A
            tt = -4.627443
        elif x3 > 5.25:
            if x14 <= -1.445:
                if x17 <= 0.045:
                    ## B
                    tt = 2.141884
                elif x17 > 0.045:
                    ## C
                    tt = 0.247894
            elif x14 > -1.445:
                ## D
                tt = -1.716035
    elif x3 > 8.345:
        if x16 <= 9.45:
            if x14 <= -0.405:
                ## E
                tt = 1.991644
            elif x14 > -0.405:
                ## F
                tt = 0.043099
        elif x16 > 9.45:
            ## G
            tt = 4.45875
        
    return tt

def my_pred_fun(row):
    beta_0 = -1.3691756
    beta = [0.00720657, -0.0214513 ,  0.05568625,  0.00303093,  0.07028513,
        -0.09168872, -0.04209721, -0.08963395, -0.01819937,  0.05159675,
         0.27044728, -0.05459181, -0.12804497, -0.06474005, -0.26164347,
         0.08821884, -0.00896532, -1.04506594,  0.20535536, -0.03090034,
         0.72036302]
    if row[1]>-99:
        exp_temp = np.exp(beta_0 + np.inner(beta,row))
        tt = exp_temp/(1+exp_temp)
    else:
        tt = 0
    return tt
    
def my_pred_fun_adjusted(row):
    beta_0 = -1.3691756
    beta = [0.00720657, -0.0214513 ,  0.05568625,  0.00303093,  0.07028513,
        -0.09168872, -0.04209721, -0.08963395, -0.01819937,  0.15,
         0.27044728, -0.05459181, -0.12804497, -0.06474005, -0.26164347,
         0.08821884, -0.00896532, -1.04506594,  0.20535536, -0.03090034,
         0.72036302]
    if row[1]>-99:
        exp_temp = np.exp(beta_0 + np.inner(beta,row))
        tt = exp_temp/(1+exp_temp)
    else:
        tt = 0
    return tt
    

# this function writes the results to a csv file
# after rearranging the data in the required format
def make_pd_frame(results,file_name):
    temp = []
    for i in range(len(results)):
        for j in range(len(results[0])):
            temp.append([x for x in results[i][j]])
    temp = pd.DataFrame(temp)
    temp = temp.transpose()
    temp.to_csv(parent_dir + "/data/pickle_test/" + file_name.split('.')[0] + '_predicted.csv',index=False)

# this function calculates the features for the given data
# it calls a separately written function from the module mf
# mf = my_function_no_keras
def feature_calc(x,ml,CP,cs):
    import implement_functions as mf
    l1 = x[0]
    l2 = x[1]
    l3 = x[2]

    # make_parallel_x applies the second check, check 2
    # it also calculates the features on the data that passes through the check
    temp_x1 = mf.make_parallel_x(l1,l2,ml,CP,cs)
    temp = temp_x1.apply(include_tree,axis=1)
    temp_x1 = pd.concat([temp_x1,temp],axis=1)
##    temp_x1.to_csv('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/featuers.csv',index=False)
    
##    temp_x2 = mf.make_parallel_x(l2,l3,ml,CP,cs)
##    temp = temp_x2.apply(include_tree,axis=1)
##    temp_x2 = pd.concat([temp_x2,temp],axis=1)
    
    # this runs the prediction on the generated features and returns the predictions
    prob1 = temp_x1.apply(my_pred_fun,axis=1)
    temp_x1 = pd.concat([temp_x1,prob1],axis=1)
    temp_x1.to_csv("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/temp_x1_output.csv",index=False)
##    prob2 = temp_x2.apply(my_pred_fun,axis=1)
##    prob3 = temp_x1.apply(my_pred_fun_adjusted,axis=1)
##    prob4 = temp_x2.apply(my_pred_fun_adjusted,axis=1)

##    prob1 = pd.DataFrame(logistic_model.predict_proba(temp_x1)).iloc[:,-1]
##    prob2 = pd.DataFrame(logistic_model.predict_proba(temp_x2)).iloc[:,-1]

    return prob1
    
# this function bunches the related thermocouples and return the bunch
# the bunch is further passed on to the feature calculating function
def make_list(window_file,active_list):
    temp = []
    for TC in active_list:
        l1 = 'TC' + str(TC)
        l2 = 'TC' + str(TC+20)
        l3 = 'TC' + str(TC+40)
        temp.append([window_file.loc[:,l1],window_file.loc[:,l2],window_file.loc[:,l3]])
    return temp


### this function calls the process and puts it into threads 
##def apply_async_TC(window_file,logistic_model,file_name):
##    import implement_functions as mf
##
##    # declaring the V1-V4 that will remain common to all
##    ML = window_file.loc[:,'M.level']
##    CS = window_file.loc[:,'C.speed']
##    CP = window_file.loc[0,'C.percent']
##    MW = window_file.loc[0,'M.width']
##    
##    # finding the active thermocouples that depend on the parameter MW (mold width)
##    # this is the first check, check 1
##    # it decides which columns need to be skipped
##    active_list = mf.find_active(MW)
##    
##    TC_dat = make_list(window_file,active_list)
##    
##    # declaring the number of parallel thread allowed
##    p_no = int(len(TC_dat)/1)
##    # calling the parallelization module and passing the required data
##    with mp.Pool(processes=p_no) as pool :
##        print("entering the parallel code, with workers = ", p_no)
##        results = [pool.apply_async(feature_calc, args=(x,ML,CP,CS,logistic_model)) for x in TC_dat]
##        output = [p.get() for p in results]
##    make_pd_frame(output,file_name)
##    return output


# Only for testing purposes
# might not have the exact functionalities of the whole program
# is not even a representative of the whole program
def apply_for_TC(window_file,file_name):
    import implement_functions as mf
    ML = window_file.loc[:,'M.level']
    CS = window_file.loc[:,'C.speed']
    CP = window_file.loc[0,'C.percent']
    MW = window_file.loc[0,'M.width']
    active_list = mf.find_active(MW)
    print(active_list)
    print(active_list[0])
    TC_dat = make_list(window_file,active_list)
    print("entering the sequential code")
    results = feature_calc(TC_dat[0],ML,CP,CS)
##    make_pd_frame([results],file_name)
##    print(results)
    return results


###############################################################################
###############################################################################
###############################################################################

# main method which calls everything else
if __name__ == '__main__':
    # importing the necessary modules
    import implement_functions as mf
    # declaring the parent directory
    global parent_dir
##    parent_dir = "/mnt2/client16/TataSteel_BDS_ProcDat/prelim_analysis"
    parent_dir = "D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis"
    # changint the current directory to the one where the data is    
    os.chdir(parent_dir + "/data/test")
    # generating a list of files on which we will iterate
    file_list = os.listdir()
    
    # taking the time
    time_start_all = time.time()
    # initializing the iteration count
    count_iter = 0
    # looping over all the files
    for file_name in file_list:
        count_iter += 1
        print(file_name,count_iter,len(file_list))
        file = pd.read_csv(file_name)
        n=file.shape[0]
##        file = file.iloc[0:n,:]

        # calling the function that does the main thing
##        start_time = time.time()
##        TC_dat = apply_async_TC(file,file_name)
##        end_time = time.time()
##        print("parallel execution time per iteration : ",(end_time-start_time)/(n-63))
##        print("total parallel execution time  : ",(end_time-time_start_all))

        start_time = time.time()
        TC_dat = apply_for_TC(file,file_name)
        end_time = time.time()
        print("parallel execution time per iteration : ",(end_time-start_time)/(n-63))
        print("total parallel execution time  : ",(end_time-time_start_all))


