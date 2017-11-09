import os
import pandas as pd

os.chdir("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/test")
temp = []

for file_name in os.listdir():
    file = pd.read_csv(file_name)
    temp.append(len(file.loc[:,'M.width'].unique()))

print(max(temp))
