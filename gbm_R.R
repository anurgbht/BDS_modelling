#relative.influence(gbm_1)
#pretty.gbm.tree(gbm_1)

setwd("D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data")
library(gbm)
library(aod)
library(ggplot2)
library(Rcpp)
library(MASS)
library(ROCR)
library(caTools)
library(caret)
library('int64')

dat <- read.csv("for_GBM.csv",na.strings = c("\\N","","NA"),sep=',')
table(dat$y_label)
names(dat)

dat<-as.data.table(dat)

# DEFINE TARGET VARIABLE

target_var <- "y_label"

#dat<-dat[sumbkt!=2]

dim(dat)

# Replace NAs with -99999
dat[is.na(dat)] <- -99999

mod_data <- dat

mod_data<-as.data.frame(dat)

mod_data$y<-mod_data$y_label

#mod_data_sub<-mod_data[ ,c("y","affluencefinalflag","ressalaryperyearofexp","hhnetworthbkt","mobilewalletbkt","insappbkt","reslevelassistmanagerflag","resbachelorflag","resleveljuniorexecflag","carsizelargeflag","resprofessfinanceflag","carsizesmallflag","disspendbkt","dthcategory","creditcardlimitbucket","age","autoinssmallflag","carownlargeflag","railavgtktsize","carownsmallflag","travelbkt")]
mod_data_sub<-mod_data[,c("y","X0","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25")]

set.seed(42)
train_rows = sample.split(mod_data_sub$y, SplitRatio=0.7)

train = mod_data_sub[ train_rows, ]

test  = mod_data_sub[!(train_rows), ]

lambda<-c(0.05,0.1,0.2)

#number of trees
n_trees=seq(from=100, to=5000, by=100)

for (i in lambda)
  
{
  #Running the gbm model
  print(i)
  boost_fit<-gbm(y~.,data=train,distribution="bernoulli",n.trees=5000,shrinkage=i,
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


