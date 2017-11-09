rm(list=ls())

library(data.table)
library(smbinning)
library(plyr)

# read the data file here, na.strings = c("\\N","","NA") this argument converts the given
# fields to NULL / NA
dat <- read.csv("for_IV.csv",na.strings = c("\\N","","NA"),sep=',')

# renaming the columns as necessary
colnames(dat) <- sapply(colnames(dat), function(x) gsub("_","", x))
colnames(dat) <- sapply(colnames(dat), function(x) gsub("//.","", x))
colnames(dat)<-c("npsqrtnpmeannpsquarett1","npmaxtt1","npmeantt1","npmintt1","npstdtt1","lastnmtt1212","lastnmtt1512","lastnmtt1712","lastnmtt1232","lastnmtt1532","lastnmtt1732","slopentt110","slopentt15","slopentt115","slopentt120","npsqrtnpmeannpsquarett2","npmaxtt2","npmeantt2","npmintt2","npstdtt2","lastnmtt2212","lastnmtt2512","lastnmtt2712","lastnmtt2232","lastnmtt2532","lastnmtt2732","slopentt210","slopentt25","slopentt215","slopentt220","npsqrtnpmeannpsquarett3","npmaxtt3","npmeantt3","npmintt3","npstdtt3","lastnmtt3212","lastnmtt3512","lastnmtt3712","lastnmtt3232","lastnmtt3532","lastnmtt3732","slopentt310","slopentt35","slopentt315","slopentt320","npsqrtnpmeannpsquarett4","npmaxtt4","npmeantt4","npmintt4","npstdtt4","lastnmtt4212","lastnmtt4512","lastnmtt4712","lastnmtt4232","lastnmtt4532","lastnmtt4732","slopentt410","slopentt45","slopentt415","slopentt420","npmeantt5","signpresenttt1tt2","Column63")
names(dat)

# checking distribution of any one varible to see that it makes sense
table(dat$Column63)

# converting the data frame to data table for further use
dat<-as.data.table(dat)

# DEFINE TARGET VARIABLE
target_var <- "Column63"

# removing unwanted data
dat<-dat[signpresenttt1tt2!=0]

dim(dat)

# Replace NAs with -99999
dat[is.na(dat)] <- -99999
summ<-smbinning.eda(dat, rounding = 3, pbar = 1)$eda   

# duplicating the data for further use
mod_data <- copy(dat)
dim(mod_data)

# BASED ON UNIQUE VALUES, CATEGORIZE VARIABLES AS CONTINUOUS OR CATEGORICAL (FACTOR)
col_names <- names(mod_data)

con_vars <- c()
cat_vars <- c()
misd_vars <- c()

for(i in seq_along(col_names))
{
  # checks the number of unique value and coerces the variable accrodingly
  # number 10 is taken as for N>10 smbinning does not allow categorical variables
  v1 <- uniqueN(mod_data[, get(col_names[i])])
  
  if(v1 > 10 && col_names[i] != target_var)
  {
    con_vars <- c(con_vars, col_names[i])
  }
  else if(v1 <= 10 && col_names[i] != target_var)
  {
    cat_vars <- c(cat_vars, col_names[i])
  }
}

# check variables before running the code further
con_vars
cat_vars

# converting the continuous variables to numeric for smbinning
for(i in seq_along(con_vars))
{
  print(con_vars[i])
  mod_data[,con_vars[i]]<-as.numeric(mod_data[,get(con_vars[i])])
  
  
}

mod_data[is.na(mod_data)] <- -99999

summ<-smbinning.eda(mod_data, rounding = 3, pbar = 1)$eda   
# CREATE EMPTY DATA FRAME FOR IV STORAGE

# declare the variables here for which you need IVs
# names should match the column names
p<-c("age","ressalaryperyearofexp","onlineshopbkt","resincome","disspendbkt","railavgtktsize","hhnetworthbkt","afflbkt","idff","autoinssmallflag","carownlargeflag","ecommtotalamountspend","cccreditlimitbkt","carownsmallflag","insappbkt","resleveljuniorexecflag","resindustrybankinsuflag","resindustryaccfinconsbusiflag","resleveltopmanagementflag","reslevelseniormanagerflag","telecomcorporateflag")

iv_chars <- data.frame("CharName" = character(0), "IV" = numeric(0), stringsAsFactors=FALSE)

# for formatting
blank_row <- c("NA", NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, "NA")
rm(iv_table)

# BASED ON VARIABLE CATEGORIZATION, APPLY BINNING FOR CATEGORICAL VARIABLES AND CALCULATE IV

for(i in seq_along(cat_vars))
{
  print(paste0("Working on categorical variable : ",i))
  set(mod_data, NULL, cat_vars[i], as.factor(mod_data[[cat_vars[i]]]))
  
  fac_bin = try(smbinning.factor(df = mod_data, y = target_var, x = cat_vars[i]), TRUE)
  
  if(class(fac_bin)=="character")
  {
    print('fac_bin error')
    misd_vars <- rbind(misd_vars,c(cat_vars[i],fac_bin))
  }
  else
  {
    print('fac_bin okay')
    iv_chars[nrow(iv_chars)+1,] <- c(cat_vars[i], fac_bin$iv) 
    df_temp <- data.frame(fac_bin$ivtable)
    df_temp$CharName <- cat_vars[i]
    
    if(!exists("iv_table")){
      iv_table <- copy(df_temp)
      iv_table <- rbind(iv_table, blank_row)
    } else{
      iv_table <- rbind(iv_table, df_temp, blank_row)
    }
    
    
  }
}

# BASED ON VARIABLE CATEGORIZATION, APPLY BINNING FOR CONTINUOUS VARIABLES AND CALCULATE IV

for(i in seq_along(con_vars))
{
  if((is.numeric(mod_data[[con_vars[i]]]) | is.integer(mod_data[[con_vars[i]]])))
  {
    
    temp_df <- mod_data[get(con_vars[i]) !=0 & get(con_vars[i]) !=-99999, .(get(con_vars[i]),get(target_var))]
    x.Pct20.Breaks=as.vector(quantile(temp_df[,1,with=F], probs=seq(0,1,0.2), na.rm=TRUE))
    Cuts.x.Pct20=x.Pct20.Breaks[1:(length(x.Pct20.Breaks))]
    cut_len <- length(Cuts.x.Pct20)
    for( j in 0:(cut_len-1))
    {
      Cuts.x.Pct20[cut_len-j+1]=Cuts.x.Pct20[cut_len-j]
    }
    Cuts.x.Pct20[1]=-99999 
    Cuts.x.Pct20[2]=0
    fac_bin=try(smbinning.custom(df=mod_data,y=target_var,x=con_vars[i],cuts=Cuts.x.Pct20))
    
    if(class(fac_bin)=="character")
    {
      misd_vars <- rbind(misd_vars,c(con_vars[i],fac_bin))
    }
    else
    {
      df_temp <- data.frame(fac_bin$ivtable)
      df_temp$CharName <- con_vars[i]
      
      iv_chars[nrow(iv_chars)+1,] <- c(con_vars[i], fac_bin$iv)
      
      if(!exists("iv_table")){
        iv_table <- copy(df_temp)
        iv_table <- rbind(iv_table, blank_row)
      } else{
        iv_table <- rbind(iv_table, df_temp, blank_row)
      }
    }
    print(paste0("Working on continuous variable : ",i))
  }
  else
  {
    misd_vars <- rbind(misd_vars,c(cat_vars[i],"not numeric / integer"))
  }
}

# this block converts the iv table to desired format and writes it to a csv

iv_table_temp <- iv_table
iv_table_temp <- subset(iv_table_temp,Cutpoint!="NA")
iv_table_temp$CharBinNum <- ave(iv_table_temp$Cutpoint, iv_table_temp$CharName, FUN = seq_along)
iv_list<-subset(iv_table,Cutpoint=="Total")
iv_list<-arrange(iv_list,desc(IV))
iv_list$IV_Char <- iv_list$IV
iv_list<-subset(iv_list, select=c("IV_Char","CharName"))
iv_table_temp2 <- merge(x=iv_table_temp, y=iv_list, by= "CharName", all.x=T)
iv_table_sorted <- iv_table_temp2[order(-as.numeric(iv_table_temp2$IV_Char), iv_table_temp2$CharName, as.numeric(iv_table_temp2$CharBinNum)),]
write.csv(iv_table_sorted,'iv_table.csv')
