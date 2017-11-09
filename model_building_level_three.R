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

setwd('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/')

data_1 = read.csv('dat_1_level_three_18_08.csv')
data_0 = read.csv('dat_0_level_three_18_08.csv')
colnames(data_0) = colnames(data_1)
data_1 = data_1[,-c(1,2,3)]
data_0 = data_0[,-c(1,2,3)]

mod_data = rbind(data_1,data_0)
mod_data$ratio = mod_data$stdv1/mod_data$stdv2
tree_model = rpart(y~.,method="class",data=mod_data,control = rpart.control(maxdepth = 30,minsplit = 25,cp = 0.01))
rpart.plot(tree_model,extra=101,digits=5,nn=FALSE,branch=0.5,cex = 0.75)
