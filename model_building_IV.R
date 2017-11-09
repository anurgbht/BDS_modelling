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

setwd('D:/Confidential/Projects/Steel/LD2 BDS/prelim_analysis/data/second iteration/constructed_data/')


data_1 = read.csv('dat_21_07_17.csv')
data_0 = read.csv('dat_zero_21_07_17.csv')

set.seed(round(runif(1,10,100)))
split <- sample.split(data_0$X0.1, SplitRatio = 9000/466218)
data_0_sample <- subset(data_0, split == TRUE)

mod_data = rbind(data_1,data_0_sample)
mod_data_all = rbind(data_1,data_0)

colnames(mod_data_all) = c("0_sopen23","1_sopen25","2_sopen27","3_signpresent120","4_signpresent122","5_signpresent127","6_astn15","7_astn25","8_astn132","9_astn232","10_np.averageCP","11_np.std1.ioc[-12:]","12_np.stdM.ioc[-12:]","13_find1peaksope12CS","14_find1peakratio12CS","15_finddrop1","16_finddrop2","17_cschangeCS","18_getkinks2","19_firstderivative2","20_highfreq","yflag")
colnames(mod_data) = c("0_sopen23","1_sopen25","2_sopen27","3_signpresent120","4_signpresent122","5_signpresent127","6_astn15","7_astn25","8_astn132","9_astn232","10_np.averageCP","11_np.std1.ioc[-12:]","12_np.stdM.ioc[-12:]","13_find1peaksope12CS","14_find1peakratio12CS","15_finddrop1","16_finddrop2","17_cschangeCS","18_getkinks2","19_firstderivative2","20_highfreq","yflag")

mod_data_all = mod_data_all[,-c(2,4,11,13,14,15)]
mod_data = mod_data[,-c(2,4,11,13,14,15)]

model = glm(yflag~.,data = mod_data, family = binomial(link = "logit"))
summary(model)

tree_model = rpart(yflag~.,method="class",data=mod_data,control = rpart.control(maxdepth = 5,minsplit = 30,cp = 0.01))
rpart.plot(tree_model,extra=101,digits=5,nn=FALSE,branch=0.5,cex = 1)


############################################################################################################
############################################################################################################
############################################################################################################

# col_names <- names(mod_data)
# target_var <- "yflag"
# con_vars <- c()
# cat_vars <- c()
# misd_vars <- c()
# mod_data<-as.data.table(mod_data)
# for(i in seq_along(col_names))
# {
#   # checks the number of unique value and coerces the variable accrodingly
#   # number 10 is taken as for N>10 smbinning does not allow categorical variables
#   v1 <- uniqueN(mod_data[, get(col_names[i])])
#   
#   if(v1 > 10 && col_names[i] != target_var)
#   {
#     con_vars <- c(con_vars, col_names[i])
#   }
#   else if(v1 <= 10 && col_names[i] != target_var)
#   {
#     cat_vars <- c(cat_vars, col_names[i])
#   }
# }
# 
# iv_chars <- data.frame("CharName" = character(0), "IV" = numeric(0), stringsAsFactors=FALSE)
# 
# # for formatting
# blank_row <- c("NA", NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, "NA")
# rm(iv_table)
# 
# for(i in seq_along(con_vars))
# {
#   if((is.numeric(mod_data[[con_vars[i]]]) | is.integer(mod_data[[con_vars[i]]])))
#   {
#     
#     temp_df <- mod_data[get(con_vars[i]) !=0 & get(con_vars[i]) !=-99999, .(get(con_vars[i]),get(target_var))]
#     x.Pct20.Breaks=as.vector(quantile(temp_df[,1,with=F], probs=seq(0,1,0.25), na.rm=TRUE))
#     Cuts.x.Pct20=x.Pct20.Breaks[1:(length(x.Pct20.Breaks))]
#     cut_len <- length(Cuts.x.Pct20)
#     for( j in 0:(cut_len-1))
#     {
#       Cuts.x.Pct20[cut_len-j+1]=Cuts.x.Pct20[cut_len-j]
#     }
#     Cuts.x.Pct20[1]=-99999 
#     Cuts.x.Pct20[2]=0
#     fac_bin=try(smbinning.custom(df=mod_data,y=target_var,x=con_vars[i],cuts=Cuts.x.Pct20))
#     
#     if(class(fac_bin)=="character")
#     {
#       misd_vars <- rbind(misd_vars,c(con_vars[i],fac_bin))
#     }
#     else
#     {
#       df_temp <- data.frame(fac_bin$ivtable)
#       df_temp$CharName <- con_vars[i]
#       
#       iv_chars[nrow(iv_chars)+1,] <- c(con_vars[i], fac_bin$iv)
#       
#       if(!exists("iv_table")){
#         iv_table <- copy(df_temp)
#         iv_table <- rbind(iv_table, blank_row)
#       } else{
#         iv_table <- rbind(iv_table, df_temp, blank_row)
#       }
#     }
#     print(paste0("Working on continuous variable : ",i))
#   }
#   else
#   {
#     misd_vars <- rbind(misd_vars,c(cat_vars[i],"not numeric / integer"))
#   }
# }
# 
# # this block converts the iv table to desired format and writes it to a csv
# 
# iv_table_temp <- iv_table
# iv_table_temp <- subset(iv_table_temp,Cutpoint!="NA")
# iv_table_temp$CharBinNum <- ave(iv_table_temp$Cutpoint, iv_table_temp$CharName, FUN = seq_along)
# iv_list<-subset(iv_table,Cutpoint=="Total")
# iv_list<-arrange(iv_list,desc(IV))
# iv_list$IV_Char <- iv_list$IV
# iv_list<-subset(iv_list, select=c("IV_Char","CharName"))
# iv_table_temp2 <- merge(x=iv_table_temp, y=iv_list, by= "CharName", all.x=T)
# iv_table_sorted <- iv_table_temp2[order(-as.numeric(iv_table_temp2$IV_Char), iv_table_temp2$CharName, as.numeric(iv_table_temp2$CharBinNum)),]
# write.csv(iv_table_sorted,'iv_table_Steel.csv')
