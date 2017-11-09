#relative.influence(gbm_1)
#pretty.gbm.tree(gbm_1)

setwd("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/constructed data/")
library(gbm)
library(aod)
library(ggplot2)
library(Rcpp)
library(MASS)
library(ROCR)
library(caTools)
library(caret)

dat <- read.csv("data_dump_24_3_17.csv",na.strings = c("\\N","","NA"),sep=',')

table(dat$X0.3)
names(dat)

target_var <- "X0.3"

#mod_data_sub<-mod_data[ ,c("y","affluencefinalflag","ressalaryperyearofexp","hhnetworthbkt","mobilewalletbkt","insappbkt","reslevelassistmanagerflag","resbachelorflag","resleveljuniorexecflag","carsizelargeflag","resprofessfinanceflag","carsizesmallflag","disspendbkt","dthcategory","creditcardlimitbucket","age","autoinssmallflag","carownlargeflag","railavgtktsize","carownsmallflag","travelbkt")]
mod_data_sub<-dat[,c("X0.3","X0","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19")]

set.seed(42)
train_rows = sample.split(mod_data_sub$X0.3, SplitRatio=0.7)

train = mod_data_sub[ train_rows, ]

test  = mod_data_sub[!(train_rows), ]

lambda<-c(0.05,0.1,0.2)
lambda = 0.05
#number of trees
n_trees=seq(from=100, to=300, by=100)
i=0.05
for (i in lambda)
  
{
  #Running the gbm model
  print(i)
  boost_fit<-gbm(X0.3~.,data=train,distribution="bernoulli",n.trees=300,shrinkage=i,
                 interaction.depth=2)
  
  # We train the model on a large number of trees and 
  #then check which combination of learning rate and no of trees works best for test
  
  #Using the model to get probabilities and performance on train data
  
  pred_train_matrix<-predict(boost_fit,train,n.trees=n_trees,type="response")
  
  dim(pred_train_matrix)
  
  train_pred<-data.frame(pred_train_matrix)
  
  pred_char_train<-sapply(train_pred,function (x){prediction(x,train$y)})
  perf_train <- lapply(pred_char_train,function(x){performance(x,"tpr","fpr")})
  ks_train<-sapply(perf_train,function(x){max(attr(x,'y.values')[[1]]-attr(x,'x.values')[[1]])},simplify=T)
  
  ks_train_list <- paste("ks_train_for_lambda", i, sep = "")
  assign(ks_train_list, ks_train)
  
  #Tuning parameters in test. We train the model on a large number of trees and 
  #then check which combination of learning rate and no of trees works best for test
  
  pred_test_matrix<-predict(boost_fit,test,n.trees=n_trees,type="response")
  
  dim(pred_test_matrix)
  
  test_pred<-data.frame(pred_test_matrix)
  
  pred_char<-sapply(test_pred,function (x){prediction(x,test$y)})
  perf <- lapply(pred_char,function(x){performance(x,"tpr","fpr")})
  ks_test<-sapply(perf,function(x){max(attr(x,'y.values')[[1]]-attr(x,'x.values')[[1]])},simplify=T)
  
  ks_test_list <- paste("ks_test_for_lambda", i, sep = "")
  boost_fit_iter <- paste("boost_fit", i, sep = "_")
  assign(ks_test_list, ks_test)
  assign(boost_fit_iter, boost_fit)
  #print(as.data.frame(relative.influence(boost_fit,n.trees=1700,sort.=TRUE)))
}

# print(as.data.frame(relative.influence(boost_fit,n.trees=3300,sort.=TRUE)))

cbind(data.frame(ks_train_for_lambda0.001),data.frame(ks_test_for_lambda0.001))
cbind(data.frame(ks_train_for_lambda0.005),data.frame(ks_test_for_lambda0.005))
cbind(data.frame(ks_train_for_lambda0.01),data.frame(ks_test_for_lambda0.01))
cbind(data.frame(ks_train_for_lambda0.05),data.frame(ks_test_for_lambda0.05))
cbind(data.frame(ks_train_for_lambda0.1),data.frame(ks_test_for_lambda0.1))
cbind(data.frame(ks_train_for_lambda0.2),data.frame(ks_test_for_lambda0.2))

print(as.data.frame(relative.influence(boost_fit,n.trees=1700,sort.=TRUE)))

# logistic regression


dat <- read.csv("data_dump_24_3_17.csv",na.strings = c("\\N","","NA"),sep=',')


set.seed(42)
dat_left = dat[,1:20]
dat_trigger = dat[,21:40]
dat_right = dat[,41:60]
y = dat[,61]
table(y)
index_file= dat[,62]

dat_left$y = y
dat_right$y = y
dat_trigger$y = y

dat_left$X19 = ifelse(dat_left$X19<0,-999,dat_left$X19)
dat_right$X19.2 = ifelse(dat_right$X19.2<0,-999,dat_right$X19.2)
dat_trigger$X19.1 = ifelse(dat_trigger$X19.1<0,-999,dat_trigger$X19.1)

library('rpart')
library('rpart.plot')
library('rattle')
fit = rpart(y~.,data=dat_trigger)
rpart.plot(fit, uniform=TRUE, main="Classification Tree for Kyphosis")



train_rows = sample.split(dat_trigger$y, SplitRatio=0.7)

train = dat_left[ train_rows, ]

test  = dat_left[!(train_rows), ]

mylogit <- glm(y~., data = train)

summary(mylogit)

