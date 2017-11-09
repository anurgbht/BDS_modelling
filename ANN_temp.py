import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn import preprocessing
import random
from sklearn.neural_network import MLPClassifier

## for keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

def give_me_data(N,r):
    theta = [random.uniform(-2*np.pi,2*np.pi) for x in range(0,N)]
    x1 = [r*np.cos(x) for x in theta]
    x2 = [r*np.sin(x) for x in theta]
    x1 = pd.DataFrame(x1)
    x2 = pd.DataFrame(x2)
    return x1,x2

################################~MAIN~##########################################

orginal_path = os.getcwd()
os.chdir('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data')

x11,x12 = give_me_data(100,2)
y1=[1 for x in range(0,100)]
data1=x11
data1['x2'] = x12
data1['y'] = y1
data1.columns= ['x1','x2','y']
x21,x22 = give_me_data(300,5)
y2=[0 for x in range(0,300)]
data2=x21
data2['x2'] = x22
data2['y'] = y2
data2.columns= ['x1','x2','y']
outdata = data1.append(data2)

plt.scatter(outdata.iloc[:,0],outdata.iloc[:,1],c = outdata.iloc[:,-1],s=120)
plt.grid(1)
plt.show()

X_all = outdata.iloc[:,:-1]
Y_all = pd.DataFrame(outdata.iloc[:,-1])
clf_keras,X_train,X_test,y_test,y_train,test_index,my_pred = prelim_keras(X_all,Y_all,0.2)
print(clf_keras.get_weights())


##temp = test_index
##temp['my_pred'] = [int(x) for x in my_pred]
##temp['y_test'] = [int(x) for x in y_test.iloc[:,0]]
##
##shit_index = pd.DataFrame()
##
##for i in range(0,len(my_pred)):
##    if (temp.iloc[i,1] == 0) & (temp.iloc[i,2] == 1):
##        print(temp.iloc[i,0])
##        shit_index = shit_index.append([temp.iloc[i,0]])
##
##print(shit_index)
##
