import os
import numpy as np
import pandas as pd
import progressbar
from tqdm import *
##################################################################################################
##################################################################################################
##################################################################################################

os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/first iteration/caster C old')
all_files = os.listdir()
##all_files = all_files[:5]
tt = []
count = 0
##pbar = progressbar.ProgressBar(maxval=len(all_files), \
##    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
for file_name in tqdm(all_files):
    file = pd.read_csv(file_name)
    cs = file.loc[:,'C.speed']
    tt.append([file_name,min(cs),max(cs),np.std(cs)])

pd.DataFrame(tt).to_csv('caster_c_recent_meta.csv',index=False)
