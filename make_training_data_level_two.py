import os
import numpy as np
import pandas as pd
import implement_functions_16_08_17 as mf
import random
import pylab as plt
from tqdm import tqdm

##################################################################################################
##################################################################################################
##################################################################################################
def global_features(temp_file,self,active_list):
    tt = []
    for TC in active_list:
        if TC != self:
            p_temp = list(temp_file.loc[:,str(TC)])
            pp = max(p_temp)
            if pp < 0:
                pp = 0
            tt.append(pp)
    q = [np.average(tt),sum(np.greater(tt,0.5)),sum(tt)]
    return q

##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/input_for_level_two_ones')
all_files = os.listdir()
tt = []
for file_name in tqdm(all_files):
    file = pd.read_csv(file_name)
    # numbering 0-19 in file
    active_list = mf.give_active_list(file.shape[1])
    self = int(file_name.split('_')[4].split('.')[0])+1
    file.columns = [str(x) for x in active_list]
    
    left = mf.find_left_list(active_list,self)
    right = mf.find_right_list(active_list,self)    

    p_self = list(file.loc[:,str(self)])
    temp = p_self[230:250]
    index = 230 + temp.index(max(temp))
    p_left = file.loc[index-15:,str(left)]
    p_right = file.loc[index-15:,str(right)]
    q = [file_name,max(temp),max(p_left),max(p_right)]
    q.extend(global_features(file.loc[index-15:index+5,:],self,active_list))
    q.append(1)
    tt.append(q)

tt = pd.DataFrame(tt)
##print(tt)
tt.to_csv('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/'+'dat_0_level_two_16_08.csv',index=False)


################################################################################################
################################################################################################
################################################################################################

##os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/input_for_level_two_zeros')
##all_files = os.listdir()
####all_files = all_files[:2]
##count = 0
##tt = []
##for file_name in all_files:
##    count += 1
##    print('Progress : ',count,len(all_files))
##    file = pd.read_csv(file_name)
##    # numbering 0-19 in file
##    active_list = mf.give_active_list(file.shape[1])
##    for self in active_list:
##        file.columns = [str(x) for x in active_list]
##        left = mf.find_left_list(active_list,self)
##        right = mf.find_right_list(active_list,self)    
##        p_self = list(file.loc[:,str(self)])
##        for index in range(15,234):
##            p_left = file.loc[index-15:index+5,str(left)]
##            p_right = file.loc[index-15:index+5,str(right)]
##            if ((p_self[index]<0.1)and(max(p_left)<0.1)and(max(p_right)<0.1)):
##                pass
##            else:
##                q = [file_name,p_self[index],max(p_left),max(p_right)]
##                q.extend(global_features(file.loc[index-15:index+5,:],self,active_list))
##                q.append(0)
##                tt.append(q)
##
##tt = pd.DataFrame(tt)
##print(tt.shape)
##tt.to_csv('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/'+'dat_0_level_two_12_08.csv',index=False)
####
