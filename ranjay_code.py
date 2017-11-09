import numpy as np
import csv

thedata = np.genfromtxt(
    'layer1.csv',
    delimiter = ',',
    dtype ='float32',
    usecols = (0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39))
a=np.array(thedata)

thedata2 = np.genfromtxt(
    'layer2.csv',
    delimiter = ',',
    dtype ='float32',
    usecols = (0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39))
b=np.array(thedata2)

thedata3 = np.genfromtxt(
    'layer3.csv',
    delimiter = ',',
    dtype ='float32',
    usecols = (0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39))
c=np.array(thedata3)

thedata4 = np.genfromtxt(
    'layer4.csv',
    delimiter = ',',
    dtype ='float32',
    usecols = (0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39))
d=np.array(thedata4)

slope1=np.zeros((3599,20))
slope2=np.zeros((3599,20))
slope3=np.zeros((3599,20))
slope4=np.zeros((3599,20))
for i in range(1,3599):
    for j in range(1,20):
        slope1[i,j]=a[i,j]-a[i-1,j]
        slope2[i,j]=b[i,j]-b[i-1,j]
        slope3[i,j]=c[i,j]-b[i-1,j]
        slope4[i,j]=d[i,j]-b[i-1,j]

count1=np.zeros((3598,20))
for i in range(1,3598):
    for j in range(1,20):
        if slope1[i,j]>0:
            count1[i,j]=count1[i-1,j]+1
        else:
            count1[i,j]=0
        
count2=np.zeros((3598,20))
for i in range(1,3598):
    for j in range(1,20):
        if slope2[i,j]>0:
            count2[i,j]=count2[i-1,j]+1
        else:
            count2[i,j]=0

count3=np.zeros((3598,20))
for i in range(1,3598):
    for j in range(1,20):
        if slope3[i,j]>0:
            count3[i,j]=count3[i-1,j]+1
        else:
            count3[i,j]=0
            
count4=np.zeros((3598,20))
for i in range(1,3598):
    for j in range(1,20):
        if slope4[i,j]>0:
            count4[i,j]=count3[i-1,j]+1
        else:
            count4[i,j]=0

h1=np.zeros((3580,20))
for i in range(20,3580):
    for j in range(1,20):
        if count1[i,j]>2 and count1[1+1,j]==0:
            h1[i,j]=a[i,j]-np.average(a[i-19:i-15,j])
        else:
            h1[i,j]=0            

h2=np.zeros((3580,20))
for i in range(20,3580):
    for j in range(1,20):
        if count2[i,j]>2 and count2[1+1,j]==0:
            h2[i,j]=b[i,j]-np.average(b[i-19:i-15,j])
        else:
            h2[i,j]=0 

h3=np.zeros((3580,20))
for i in range(20,3580):
    for j in range(1,20):
        if count3[i,j]>2 and count3[1+1,j]==0:
            h3[i,j]=c[i,j]-np.average(c[i-19:i-15,j])
        else:
            h3[i,j]=0

h4=np.zeros((3580,20))
for i in range(20,3580):
    for j in range(1,20):
        if count4[i,j]>2 and count4[1+1,j]==0:
            h4[i,j]=d[i,j]-np.average(d[i-20:i-15,j])
        else:
            h4[i,j]=0
            
foutput=[]
for i in range(20,3580):
    for j in range(1,20):
        if ((count2[i,j]>2) and (count2[i+1,j]==0)):
            h1max = max(h1[(i-20):(i-5),j])
            print (i,j,count2[i,j],h2[i,j],h1max)
            for m in range ((i-20),(i-5)):
                if h1[m,j]==h1max:
                    s=m
            temp=b[i,j]-a[s,j]
            print (i,j,count2[i,j],h2[i,j],h1max,s,count1[s,j],temp)
            outp = [i,j,count2[i,j],h2[i,j],h1max,s,count1[s,j],temp]
            foutput.append(outp)

with open('test.csv', 'w', newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(foutput)
