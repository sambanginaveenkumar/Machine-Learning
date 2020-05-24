
library(xgboost)
library(readr)
library(stringr)
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggord)
library(ggplot2)
library(Hmisc)
library(klaR)
library(klaR)
library(MASS)
library(nnet)
library(plyr)
library(pROC)
library(psych)
library(scatterplot3d)
library(SDMTools)
library(dplyr)
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)
library(gbm)

setwd("E:/r direct/Machine Learning/Assignment")

cars<-read.csv("Cars.csv",header=TRUE)

##########BOOSTING WITH CLASSIFICATION
# Partition data
library(rminer)
library(e1071)
library(class)

colnames(cars)[colSums(is.na(cars)) > 0]
length(which(is.na(cars$MBA)))
which(is.na(cars$MBA))
cars[145,]
table(cars$MBA)
cars$MBA[145] <- 0


cars$Female<-ifelse(cars$Gender=="Female",1,0)


##library(caret)

# Set the seed for the purpose of reproducible results
set.seed(36756)

pd<-sample(2,nrow(cars),replace=TRUE, prob=c(0.7,0.3)) ### FOR dividing into 2 parts
train<-cars[pd==1,]
test<-cars[pd==2,]

c(nrow(train), nrow(test))
prop.table(table(train$Transport))
prop.table(table(test$Transport))


#--------- normalize the required fields (that is non-binary numeric/integer fields) --------------------------
# 
# normalize<-function(x){
#   +return((x-min(x))/(max(x)-min(x)))}
# 
# train<-transform(train, Age=ave(train$Age,FUN = normalize))
# train<-transform(train, Work.Exp=ave(train$Work.Exp,FUN = normalize))
# train<-transform(train, Salary=ave(train$Salary,FUN = normalize))
# train<-transform(train, Distance=ave(train$Distance,FUN = normalize))
# 
# test<-transform(test, Age=ave(test$Age,FUN = normalize))
# test<-transform(test, Work.Exp=ave(test$Work.Exp,FUN = normalize))
# test<-transform(test, Salary=ave(test$Salary,FUN = normalize))
# test<-transform(test, Distance=ave(test$Distance,FUN = normalize))

str(train)
str(test)


###Now Decision Trees
set.seed(79879)
DT<-rpart(Transport~., method="class",train)
rpart.plot(DT)
rpart.plot(DT, type=3,extra=101,fallen.leaves = T)
summary(DT)


pred.DT= predict(DT, type = "class",test)
tabDT<-table( test$Transport,pred.DT)
tabDT
accuracy.DT<-sum(diag(tabDT))/sum(tabDT)
accuracy.DT
waste.DT<-tabDT[1,2]/(tabDT[1,2]+tabDT[2,2])
waste.DT
lost.DT<-tabDT[2,1]/(tabDT[2,1]+tabDT[2,2])
lost.DT
Loss.fn.DT<-0.6*(lost.DT)^2+0.4*(waste.DT)^2
Loss.fn.DT
############NOW RANDOM FOREST
set.seed(79879)

#Run RFM
train$Transport<-as.factor(train$Transport)
test$Transport<-as.factor(test$Transport)


rfm<-randomForest(Transport~.,train)

predictrfm<-predict(rfm,test)
tab.rfm<-table(test$Transport,predictrfm)
tab.rfm
accuracy.rfm<-sum(diag(tab.rfm))/sum(tab.rfm)
accuracy.rfm
waste.rfm<-tab.rfm[1,2]/(tab.rfm[1,2]+tab.rfm[2,2])
waste.rfm
lost.rfm<-tab.rfm[2,1]/(tab.rfm[2,1]+tab.rfm[2,2])
lost.rfm
Loss.fn.rfm<-0.6*(lost.rfm)^2+0.4*(waste.rfm)^2
Loss.fn.rfm

str(train)

train_labs <- as.numeric(train$Transport) - 1
val_labs <- as.numeric(test$Transport) - 1

new_tr <- model.matrix(~.+0, data = train[, -9])
new_ts <- model.matrix(~.+0,data = test[,-9])

dtrain <- xgb.DMatrix(data = new_tr,label = train_labs) 
dtest <- xgb.DMatrix(data = new_ts,label = val_labs) 


params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 3, eval_metric = "mlogloss")

# Calculate # of folds for cross-validation
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 1000, nthread = 10,nfold = 50, showsd = TRUE, 
                stratified = TRUE, print.every.n = 10, early_stop_round = 20, maximize = FALSE, prediction = TRUE)

library(dplyr)
# Function to compute classification error
classification_error <- function(conf_mat) {
  conf_mat = as.matrix(conf_mat)
  
  error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
  
  return (error)
}

# Mutate xgb output to deliver hard predictions
xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"), label = train_labs + 1)

# Examine output
head(xgb_train_preds)

xgb_conf_mat <- table(true = train_labs + 1, pred = xgb_train_preds$max)
cat("XGB Training Classification Error 
    Rate:", classification_error(xgb_conf_mat), "\n")

xgb_conf_mat_2 <- confusionMatrix(factor(xgb_train_preds$label),
                                  factor(xgb_train_preds$max),
                                  mode = "everything")

print(xgb_conf_mat_2)

xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)

# Predict for validation set
xgb_val_preds <- predict(xgb_model, newdata = dtest)

xgb_val_out <- matrix(xgb_val_preds, nrow = 3, ncol = length(xgb_val_preds) / 3) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"), label = val_labs + 1) 

# Confustion Matrix
xgb_val_conf <- table(true = val_labs + 1, pred = xgb_val_out$max)

cat("XGB Validation Classification Error Rate:", classification_error(xgb_val_conf), "\n")

xgb_val_conf2 <- confusionMatrix(factor(xgb_val_out$label),
                                 factor(xgb_val_out$max),
                                 mode = "everything")

print(xgb_val_conf2)
