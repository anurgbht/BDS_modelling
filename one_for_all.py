import os
import numpy as np
import pandas as pd
import BDS_module_17_10_17 as bds
import random
import pylab as plt
from tqdm import tqdm
import time

##################################################################################################
##################################################################################################
##################################################################################################
def extract_features(temp):
    q = temp[1:-1].split(',')
    prob = float(q[-4])
    if prob > 0.1:
        TC = float(q[-1])
        time = float(q[-3])
        cross_far = float(q[-5])
        cross_near = float(q[-6])
        cross = float(q[-7])
        long_term_l1_rise = float(q[-8])
        long_term_l2_rise = float(q[-9])
        long_term_l1_drop = float(q[-10])
        far_amp_l2 = float(q[-11])
        far_std_l2 = float(q[-12])
        far_amp_l1 = float(q[-13])
        far_std_l1 = float(q[-14])
        sudd_drop = float(q[-15])
        imm_slope_ml = float(q[-16])
        imm_slope_l1 = float(q[-17])
        drop_time_l2 = float(q[-18])
        rise_time_l1 = float(q[-19])
        high_freq = float(q[-20])
        ml = float(q[-21])
        cs = float(q[-22])
        l_avg = float(q[-23])
        l2_drop = float(q[-24])
        l1_drop = float(q[-25])
        win_std2 = float(q[-28])
        win_std1 = float(q[-29])

        tt = [time,TC,prob,win_std1,win_std2,l1_drop,l2_drop,cross,cross_near,cross_far,l_avg,cs,ml,high_freq,drop_time_l2,rise_time_l1,imm_slope_l1,imm_slope_ml,sudd_drop,long_term_l1_rise,long_term_l2_rise,long_term_l1_drop,far_amp_l2,far_std_l2,far_amp_l1,far_std_l1]
    else:
        tt = []
    return tt

def extract_raw_features(temp):
    q = temp[1:-1].split(',')
    tt = [float(x) for x in q[:10]]
    return tt

def extract_prob(temp):
    q = temp[1:-1].split(',')
    prob = float(q[-4])
    return prob

def get_neighbor_features(temp_file,self):
    all_size = temp_file.shape[1]
    if temp_file.shape[0]>15:
        temp_file = temp_file.iloc[-15:,:]
    self_prob = extract_prob(temp_file.iloc[-1,self])
    left1 =  list(temp_file.iloc[:,get_left(all_size,self)])
    left1 = [extract_prob(x) for x in left1]
    left2 =  list(temp_file.iloc[:,get_left(all_size,get_left(all_size,self))])
    left2 = [extract_prob(x) for x in left2]
    
    right1 =  list(temp_file.iloc[:,get_right(all_size,self)])
    right1 = [extract_prob(x) for x in right1]
    right2 =  list(temp_file.iloc[:,get_right(all_size,get_right(all_size,self))])
    right2 = [extract_prob(x) for x in right2]
    neighbor = max([max(left1),max(left2),max(right1),max(right1)])

    tt = [neighbor, self_prob,find_opposite(self,all_size,temp_file)]
    return tt

def is_not_in_future(temp_file_future,self):
    all_size = temp_file_future.shape[1]
    left =  list(temp_file_future.iloc[:,get_left(all_size,self)])
    left = [extract_prob(x) for x in left]
    right =  list(temp_file_future.iloc[:,get_right(all_size,self)])
    right = [extract_prob(x) for x in right]
    neighbor = max([max(left),max(right)])
    return (neighbor < 0.1)

def get_left(all_size,self):
    if self == 0:
        left = all_size-1
    else:
        left = self-1
    return left

def get_right(all_size,self):
    if self == all_size-1:
        right = 0
    else:
        right = self+1
    return right

def find_opposite(self,all_size,temp_file):
    if all_size > 15:
        tt = [self+(all_size/2)-2,self+(all_size/2)-1,self+(all_size/2),self+(all_size/2)+1,self+(all_size/2)+2]
    else:
        tt = [self+(all_size/2)-1,self+(all_size/2),self+(all_size/2)+1]
    opp = []
    for i in tt:
        if i >= all_size:
            opp.append(i-all_size)
        elif i < 0:
            opp.append(i+all_size-1)
        else:
            opp.append(i)
    # finding the history of all the opposites
    opp = [int(x) for x in opp]
    opp_prob = []
    for i in opp:
        temp = list(temp_file.iloc[:,i])
        temp = [extract_prob(x) for x in temp]
        opp_prob.append(max(temp))
    return max(opp_prob)

def my_filter121(row):
    tt = ((row["self"] > 0.4) and
          (row['cross']>0.4) and
          (row['drop slope l1'] > 0) and
          ((row["ratio"] > 1.4) or (row["neighbor"] > 0.1)) and
          (((row["cross_near"] < 0) and (row["cross_near"]-row["cross_far"] > 14)) or (row["cross_near"] >= 0)) and
          (row["immediate slope l1"] < 5) and
          (row["rise time l2"] < 35) and
          (((row["cross_far"] < 0) and (row["rise amp l2"] - row["rise amp l1"] > -2)) or (row["cross_far"] >= 0)) and
          (((row["cs change"] > 0.15) and (row["time delta"] < 7)) or (row["cs change"] <= 0.15)) and
          ((((row["immediate slope l1"] < 5) and (row["immediate slope l1"] > -2)) and (row["sudden drop l1"] > 3)) or (row["immediate slope l1"] <= -2)))    
    return tt

def my_filter122(row):
    tt = ((row["l1 avg"] < 120) and
          (row['cross']>1) and
          (row['drop slope l1'] > 0) and
          (row["ratio"] > 2) and
          (((row["cross_near"] < 0) and (row["cross_near"]-row["cross_far"] > 14)) or (row["cross_near"] >= 0)) and
          (row["immediate slope l1"] < 2) and
          (row["rise time l2"] < 35) and
          (((row["cs change"] > 0.15) and (row["time delta"] < 7)) or (row["cs change"] <= 0.15)) and
          ((((row["immediate slope l1"] < 5) and (row["immediate slope l1"] > -2)) and (row["sudden drop l1"] > 3)) or (row["immediate slope l1"] <= -2)))        
    return tt

def my_filter231(row):
    tt = ((row["self"] > 0.4) and
          (row["rise amp l1"] > 3) and
          (row['cross']>0.4) and
          (row['drop slope l1'] > 0) and
          ((row["ratio"] > 1.4) or (row["neighbor"] > 0.1)) and
          (row["cross_near"] >= 0) and
          (row["immediate slope l1"] < 5) and
          (row["rise time l2"] < 35) and
          (((row["cross_far"] < 0) and (row["rise amp l2"] - row["rise amp l1"] > -2)) or (row["cross_far"] >= 0)) and
          (((row["cs change"] > 0.15) and (row["time delta"] < 10)) or (row["cs change"] <= 0.15)) and
          ((((row["immediate slope l1"] < 5) and (row["immediate slope l1"] > -2)) and (row["sudden drop l1"] > 3)) or (row["immediate slope l1"] <= -2)))    
    return tt

##(((row["cross_near"] < 0) and (row["cross_near"]-row["cross_far"] > 14)) or (row["cross_near"] >= 0)) and

def remove_duplicate(dat):
    tt = []
    all_files = list(set(dat.loc[:,'file name']))
    all_files.sort()
    for file in all_files:
        temp_file = dat.loc[dat.loc[:,'file name'] == file,:].reset_index().drop('index',1)
        tt.append(list(temp_file.iloc[0,:]))
        for i in range(temp_file.shape[0]-1):
            if temp_file.loc[i+1,'time'] > temp_file.loc[i,'time'] + 60:
                tt.append(list(temp_file.iloc[i+1,:]))
    return pd.DataFrame(tt)

def write_file(extracted,name,write_tag):
    filter1 = extracted.loc[extracted.iloc[:,-1],:]
    filter1 = filter1.reset_index().drop('index',1)
    print(filter1.shape)

    filter2 = remove_duplicate(filter1)
    filter2.columns = filter1.columns
    print(filter2.shape)

    if write_tag == 1:
        name = name +'_' + time.strftime("%d_%m") + '.xlsx'
        writer = pd.ExcelWriter(name)
        extracted.to_excel(writer,sheet_name = 'raw data')
        filter1.to_excel(writer,sheet_name = 'filtered data')
        filter2.to_excel(writer,sheet_name = 'duplicates removed')
        writer.save()
        print(os.getcwd())
    return filter1,filter2

def extract_all(path,tag):
    os.chdir(path)
    all_files = os.listdir()
    ##all_files = all_files[:10]
    extracted=[]
    for file_name in tqdm(all_files):
##        print(file_name)
        file = pd.read_csv(file_name)
##        print(file.shape)
        for i in range(file.shape[0]-10):
            temp = list(file.iloc[i,:])
            for j in range(len(temp)):
                temp_extract = extract_features(temp[j])
                if len(temp_extract)> 0:
                    temp_raw = extract_raw_features(temp[j])
                    temp_file = file.iloc[:i+1,:]
    ##                temp_file_future = file.iloc[i:i+20,:]
    ##                if is_not_in_future(temp_file_future,TC_index):
                    neighbor_features = get_neighbor_features(temp_file,j)
                    temp_extract = [file_name.replace('_level_three.csv','')] + temp_extract + list(neighbor_features) + temp_raw
                    extracted.append(temp_extract)

    extracted = pd.DataFrame(extracted)
    extracted.columns = ["file name","time","TC","self","std1","std2","l1 drop","l2 drop","cross2","cross_near","cross_far","l1 avg","cs change","ml change","high freq","drop_time_l2","rise_time_l1","immediate slope l1","immediate slope ml","sudden drop l1","long_term_l1_rise","long_term_l2_rise","long_term_l1_drop","far_amp_l2","far_std_l2","far_amp_l1","far_std_l1","neighbor","self2","opp","rise slope l1","rise slope l2","cross","rise amp l1","rise amp l2","drop slope l1","drop amp l1","rise time l2","drop time l1","time delta"]
    extracted.loc[:,'ratio'] = extracted.loc[:,'std1']/extracted.loc[:,'std2']
    extracted.loc[:,'layer_tag'] = tag
    return extracted

def main_loop_extract(path12,path23,name1,name2,name3,write_tag):
    print("This is the main loop extract. Running on new files.")
    extracted12 = extract_all(path12,12)
    extracted23 = extract_all(path23,23)

    if write_tag == 1:
        extracted12.to_csv(name1,index=False)
        extracted23.to_csv(name2,index=False)

    ef121 = extracted12.loc[extracted12.loc[:,"self"] > 0.4,:].reset_index().drop('index',1)
    ef122 = extracted12.loc[extracted12.loc[:,"self"] <= 0.4,:].reset_index().drop('index',1)

    ef121 = pd.concat([ef121, ef121.apply(my_filter121,axis=1)],axis=1)
    ef122 = pd.concat([ef122, ef122.apply(my_filter122,axis=1)],axis=1)
    extracted23 = pd.concat([extracted23, extracted23.apply(my_filter231,axis=1)],axis=1)

    print(ef121.shape,ef122.shape,extracted23.shape)
    extracted = pd.concat([ef121,ef122,extracted23],axis=0)
    print(extracted.shape,ef121.shape,ef122.shape,extracted23.shape)
    extracted = extracted.sort_values(by=['file name','time','TC'],ascending=[1,1,1]).reset_index().drop('index',1)

    filter1,filter2 = write_file(extracted,name3,write_tag)
    return extracted,filter1,filter2

def main_loop_filter(path_construct,name1,name2,name3,write_tag):
    print("This is the main loop filter. Running on files already written.")
    os.chdir(path_construct)
    extracted12 = pd.read_csv(name1)
    extracted23 = pd.read_csv(name2)
    
    ef121 = extracted12.loc[extracted12.loc[:,"self"] > 0.4,:].reset_index().drop('index',1)
    ef122 = extracted12.loc[extracted12.loc[:,"self"] <= 0.4,:].reset_index().drop('index',1)

    ef121 = pd.concat([ef121, ef121.apply(my_filter121,axis=1)],axis=1)
    ef122 = pd.concat([ef122, ef122.apply(my_filter122,axis=1)],axis=1)
    extracted23 = pd.concat([extracted23, extracted23.apply(my_filter231,axis=1)],axis=1)

    print(ef121.shape,ef122.shape,extracted23.shape)
    extracted = pd.concat([ef121,ef122,extracted23],axis=0)
    print(extracted.shape,ef121.shape,ef122.shape,extracted23.shape)
    extracted = extracted.sort_values(by=['file name','time','TC'],ascending=[1,1,1]).reset_index().drop('index',1)

    filter1,filter2 = write_file(extracted,name3,write_tag)
    return extracted,filter1,filter2
    
##################################################################################################
##################################################################################################
##################################################################################################
def make_all(path_read,path_save,tag):
    os.chdir(path_read)
    all_files = os.listdir()
    ##all_files = all_files[:3]
    count = 0
    all_start = time.time()
    for file_name in all_files:
        start = time.time()
        count += 1
        print(count,len(all_files))
        file = pd.read_csv(file_name)
        mw = file.loc[0,'M.width']
        active_list = bds.find_active(mw)
        ml  = file.loc[:,'M.level']
        cs = file.loc[:,'C.speed']
        prob_temp = []
        for i in range(10,file.shape[0]):
            if i<60:
                temp = file.iloc[:i,:]
            else:
                temp = file.iloc[i-60:i,:]
            prob_temp.append(bds.prob_level_three(temp,active_list,i,file_name,tag))
        prob_temp = pd.DataFrame(prob_temp)
    ##    print(prob_temp)
        prob_temp.to_csv(path_save+file_name.split('.')[0]+'_level_three.csv',index=False)

##################################################################################################
##################################################################################################
##################################################################################################

path_read = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test'
path_save = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test pred 12/'
make_all(path_read,path_save,12)

path_save = 'D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test pred 23/'
make_all(path_read,path_save,23)

#####################################################################################
##################################################################################################
##################################################################################################
write_flag = 0

path12 = "D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test pred 12"
path23 = "D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/test pred 23"
name1 = 'test_12.csv'
name2 = 'test_23.csv'
name3 = 'test_level_three_'
extracted,filter1,filter2 = main_loop_extract(path12,path23,name1,name2,name3,write_flag)
wait = input('satisfied ?')

##################################################################################################
##################################################################################################
##################################################################################################

##path_construct = "D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data"
##name1 = 'caster_c_old_12.csv'
##name2 = 'caster_c_old_23.csv'
##name3 = 'caster_c_old_level_three_23_10'
##write_tag = 0
##filter2_b = main_loop_filter(path_construct,name1,name2,name3,write_tag)
