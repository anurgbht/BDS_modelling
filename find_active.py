import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import my_functions as mf

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/new_dat')
all_files = os.listdir()

count = 0
temp = pd.DataFrame()
missed = 0
for file_name in all_files:
    count = count + 1
    print(count,len(all_files))
    if file_name.split('_')[3] == '2':
        file = pd.read_csv(file_name)
        therm_list = mf.find_active((file.loc[0,'M.width']))
        try:
            therm_list.index(file.TC_layer[0])
            flag = 1
        except:
            missed = missed + 1
            flag = 0
        temp = temp.append([[file_name,flag,file.TC_layer[0],(file.loc[0,'M.width']/2),therm_list]])

print(temp.shape,missed)
temp.to_csv('missed_files.csv')
