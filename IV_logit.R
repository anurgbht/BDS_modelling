library(data.table)
library(smbinning)
library(plyr)
library(gbm)
library(aod)
library(ggplot2)
library(Rcpp)
library(MASS)
library(ROCR)
library(caTools)
library(caret)

data = read.csv('D://Confidential//Projects//Steel/LD2 BDS/prelim_analysis/data/constructed data/data_logit.csv')
mod_data = data[,c(17,18)]

mod_data$X18 = sapply(mod_data$X17,function(x) ifelse(x=='C','D',as.character(x)))
mod_data$X19 = sapply(mod_data$X18,function(x) as.factor(x))

fac_bin = smbinning.factor(df = mod_data, y = 'y', x = 'X19')
fac_bin$ivtable$WoE

temp = sapply(mod_data$X19,function(x) ifelse(x == 'A',-0.4546,
                                       ifelse(x=='B',2.6383,
                                              ifelse(x=='D',-4.8598,
                                                     ifelse(x=='E',0.2385,
                                                            ifelse(x=='F',1.8404,
                                                                   ifelse(x=='G',4.0298,
                                                                          ifelse(x=='H',0.6753,-9999))))))))
data$X17 = temp

write.csv(x = data,file = 'for_python_logit.csv',row.names = FALSE)

train_rows = sample.split(data$y, SplitRatio=0.7)
train = data[ train_rows, ]
test  = data[!(train_rows), ]

model <- glm(y ~.,family=binomial(link='logit'),data=train)
summary(model)

