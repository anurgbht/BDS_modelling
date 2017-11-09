rm(list=ls())

library(data.table)
library(smbinning)
library(plyr)
library(caTools)
library(glmnet)
library(glm)
library(glm2)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(glmulti)
library(partykit)

setwd('D:/OneDrive - Tata Insights and Quants, A division of Tata Industries/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/')

dat = read.csv('for_level_2_05_10.csv')
dat = dat[,-c(1,2,3)]

tree_model = rpart(tag~.,method="class",data=dat,control = rpart.control(maxdepth = 30,minsplit = 15,cp = 0.01))
rpart.plot(tree_model,extra=101,digits=5,nn=FALSE,branch=0.5,cex = 1)
