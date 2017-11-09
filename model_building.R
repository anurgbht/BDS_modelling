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

data_1 = read.csv('test_dat_16_08_17.csv')
data_0 = read.csv('test_dat_zero_16_08_17.csv')
data_1 = data_1[,-c(1)]
data_0 = data_0[,-c(1)]


tt = model$coefficients

# for(i in 1:1000){
print(i)
t_seed = round(runif(1,1,100000))
set.seed(t_seed)
split = sample.split(data_0$X0.1, SplitRatio = 5000/dim(data_0)[1])
data_0_sample = subset(data_0, split == TRUE)

mod_data_all = rbind(data_1,data_0_sample)

mod_data = mod_data_all
# mod_data$cat = ifelse(mod_data$X4 < 13.7,-4.91,
#                       ifelse(mod_data$X0<0.55273,.77177,
#                              ifelse(mod_data$X2,0.603,5.121555)))
model = glm2(X0.1 ~ .,data = mod_data, family = binomial(link = "logit"),control = glm.control(epsilon = 1e-8, maxit = 2000, trace = FALSE))
# tt = cbind(tt,model$coefficients)
# }
summary(model)

tree_model = rpart(X0.1~.,method="class",data=mod_data,control = rpart.control(maxdepth = 30,minsplit = 30,cp = 0.01))
rpart.plot(tree_model,extra=101,digits=5,nn=FALSE,branch=0.5,cex = 1)
print(t_seed)
