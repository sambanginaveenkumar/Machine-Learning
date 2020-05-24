
setwd("E:/r direct/Machine Learning/Assignment")

cars<-read.csv("Cars.csv",header=TRUE)


#----- check the structure of the data
str(cars)
summary(cars)
table(cars$Transport)
(61/444)*100


#----- check for missing values
library(naniar)
gg_miss_var(cars,show_pct=TRUE)

colnames(cars)[colSums(is.na(cars)) > 0]
length(which(is.na(cars$MBA)))
which(is.na(cars$MBA))
cars[145,]
table(cars$MBA)
cars$MBA[145] <- 0


#----- check for zero variance

library(caret)
nzv <- nearZeroVar(cars)
nzv
names(cars)[nzv]


# --------------outlier detection - univariate box plot analysis -------------------

library(plyr)
options(scipen=999)
options(max.print=999999)

box<- boxplot(cars$Age,col = "Skyblue",xlab="Boxplot For Age",horizontal = TRUE)
x<-which(cars$Age %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers

table(box$out)

box<- boxplot(cars$Engineer,col = "Skyblue",xlab="Boxplot For Engineer",horizontal = TRUE)
x<-which(cars$Engineer %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers
table(box$out)

box<- boxplot(cars$MBA,col = "Skyblue",xlab="Boxplot For MBA",horizontal = TRUE)
x<-which(cars$MBA %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers
table(box$out)

box<- boxplot(cars$Salary,col = "Skyblue",xlab="Boxplot For Salary",horizontal = TRUE)
x<-which(cars$Salary %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers
table(box$out)

box<- boxplot(cars$Distance,col = "Skyblue",xlab="Boxplot For Distance",horizontal = TRUE)
x<-which(cars$Distance %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers
table(box$out)

box<- boxplot(cars$license,col = "Skyblue",xlab="Boxplot For license",horizontal = TRUE)
x<-which(cars$license %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers
table(box$out)

box<- boxplot(cars$Work.Exp,col = "Skyblue",xlab="Boxplot For Work.Exp",horizontal = TRUE)
x<-which(cars$Work.Exp %in% box$out)   # returns the row numbers of outliers
table(cars$Transport[x])            # check the Transport mode of those outlier records
length(box$out)    # number of outliers
table(box$out)


###--------- multivariate analysis corr plot -----------#### age work exp salary are highly correlated

cars1<-cars
library(corrplot)
for(i in 1:ncol(cars1)){
  cars1[,i] <- as.integer(cars1[,i])
}
corrplot(corrplot(cor(cars1)), na.label = " ")


# ----------------------- define some dummies ---------------------------

cars$Female<-ifelse(cars$Gender=="Female",1,0)

# cars$Two_Wheeler<-ifelse(cars$Transport=="2Wheeler",1,0)
# cars$Car<-ifelse(cars$Transport=="Car",1,0)
# cars$Public_Transport<-ifelse(cars$Transport=="Public Transport",1,0)

str(cars)

## ---------------------------- split the data into train(70%) and test(30%) ------------------------

library(caret)

# Set the seed for the purpose of reproducible results
set.seed(36756)

pd<-sample(2,nrow(cars),replace=TRUE, prob=c(0.7,0.3)) ### FOR dividing into 2 parts
train<-cars[pd==1,]
test<-cars[pd==2,]

c(nrow(train), nrow(test))
prop.table(table(train$Car))
prop.table(table(test$Car))


#--------- normalize the required fields (that is non-binary numeric/integer fields) --------------------------

normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}

train<-transform(train, Age=ave(train$Age,FUN = normalize))
train<-transform(train, Work.Exp=ave(train$Work.Exp,FUN = normalize))
train<-transform(train, Salary=ave(train$Salary,FUN = normalize))
train<-transform(train, Distance=ave(train$Distance,FUN = normalize))

test<-transform(test, Age=ave(test$Age,FUN = normalize))
test<-transform(test, Work.Exp=ave(test$Work.Exp,FUN = normalize))
test<-transform(test, Salary=ave(test$Salary,FUN = normalize))
test<-transform(test, Distance=ave(test$Distance,FUN = normalize))

str(train)
str(test)
train$Car<-as.factor(train$Car)
test$Car<-as.factor(test$Car)
str(train)
str(test)

#---------------------------- KNN model  ------------------------------------

train_final <- train[,c(9,1,3:8,10)]
test_final <- test[,c(9,1,3:8,10)]
str(train_final)  
str(test_final)
library(class)
car_pred1<-knn(train=train_final[,-1],test=test_final[,-1], cl=train_final[,1],k=3)
tab.knn1<-table(test_final[,1],car_pred1)
tab.knn1

accuracy.knn1<-sum(diag(tab.knn1))/sum(tab.knn1)
accuracy.knn1 ### 76.76

#-----------------------KNN Model after removing the work EXP and Salary as these two were multi collinear with age variable----
train_final <- train[,c(9,1,3,4,7,8,10)]
test_final <- test[,c(9,1,3,4,7,8,10)]
str(train_final)  
str(test_final)
library(class)
car_pred2<-knn(train=train_final[,-1],test=test_final[,-1], cl=train_final[,1],k=3)
tab.knn2<-table(test_final[,1],car_pred2)
tab.knn2

accuracy.knn2<-sum(diag(tab.knn2))/sum(tab.knn2)
accuracy.knn2  ### 78.87

#--------------------------- Naive Bayes model ---------------------------
train_final <- train[,c(9,1,3:8,10)]
test_final <- test[,c(9,1,3:8,10)]

library(rminer)
library(e1071)
NB1<-naiveBayes(x=train_final[,-1], y=train_final$Transport)
y_pred1<-predict(NB1,newdata=test_final[-1])
str(y_pred1)

#Confusion matrix
cm.NB1=table(test_final[,1],y_pred1)
cm.NB1
accuracy.NB1<-sum(diag(cm.NB1))/sum(cm.NB1)
accuracy.NB1   ###  76.05

#--------------------------- Naive Bayes model after removing the work EXP and Salary as these two were multi collinear with age variable ---------------------------
train_final <- train[,c(9,1,3,4,7,8,10)]
test_final <- test[,c(9,1,3,4,7,8,10)]

library(rminer)
library(e1071)
NB2<-naiveBayes(x=train_final[,-1], y=train_final$Transport)
y_pred2<-predict(NB2,newdata=test_final[-1])
str(y_pred2)

#Confusion matrix
cm.NB2=table(test_final[,1],y_pred2)
cm.NB2
accuracy.NB2<-sum(diag(cm.NB2))/sum(cm.NB2)
accuracy.NB2  #### 76.76

#-----Prediction for 2 records KNN -----
car_test = read.csv(file.choose(),header = TRUE)

dim(car_test)

cars$Female<-ifelse(cars$Gender=="Female",1,0)
normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}

car_test<-transform(car_test, Age=ave(train$Age,FUN = normalize))
car_test<-transform(car_test, Work.Exp=ave(train$Work.Exp,FUN = normalize))
car_test<-transform(car_test, Salary=ave(train$Salary,FUN = normalize))
car_test<-transform(car_test, Distance=ave(train$Distance,FUN = normalize))

str(car_test)
car_test_final <- car_test[,c(1:8)]

car_test$PredictedTransport <- predict(car_pred1, car_test)
car_test







































