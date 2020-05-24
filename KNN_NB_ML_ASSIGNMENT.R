setwd("E:/r direct/Machine Learning/Assignment")

cars<-read.csv("Cars.csv",header=TRUE)

library(rminer)
library(e1071)
library(class)

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

# library(caret)
# nzv <- nearZeroVar(cars)
# nzv
# names(cars)[nzv]


# --------------outlier detection - univariate box plot analysis -------------------

# library(plyr)
# options(scipen=999)
# options(max.print=999999)
# 
# box<- boxplot(cars$Age,col = "Skyblue",xlab="Boxplot For Age",horizontal = TRUE)
# x<-which(cars$Age %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)
# 
# box<- boxplot(cars$Engineer,col = "Skyblue",xlab="Boxplot For Engineer",horizontal = TRUE)
# x<-which(cars$Engineer %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)
# 
# box<- boxplot(cars$MBA,col = "Skyblue",xlab="Boxplot For MBA",horizontal = TRUE)
# x<-which(cars$MBA %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)
# 
# box<- boxplot(cars$Salary,col = "Skyblue",xlab="Boxplot For Salary",horizontal = TRUE)
# x<-which(cars$Salary %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)
# 
# box<- boxplot(cars$Distance,col = "Skyblue",xlab="Boxplot For Distance",horizontal = TRUE)
# x<-which(cars$Distance %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)
# 
# box<- boxplot(cars$license,col = "Skyblue",xlab="Boxplot For license",horizontal = TRUE)
# x<-which(cars$license %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)
# 
# box<- boxplot(cars$Work.Exp,col = "Skyblue",xlab="Boxplot For Work.Exp",horizontal = TRUE)
# x<-which(cars$Work.Exp %in% box$out)   # returns the row numbers of outliers
# table(cars$Transport[x])            # check the Transport mode of those outlier records
# length(box$out)    # number of outliers
# table(box$out)


###--------- multivariate analysis corr plot -----------#### age work-exp salary are highly correlated
# 
# cars1<-cars
# library(corrplot)
# for(i in 1:ncol(cars1)){
#   cars1[,i] <- as.integer(cars1[,i])
# }
# corrplot(corrplot(cor(cars1)), na.label = " ")


# ----------------------- define some dummies ---------------------------

cars$Female<-ifelse(cars$Gender=="Female",1,0)

# cars$Two_Wheeler<-ifelse(cars$Transport=="2Wheeler",1,0)
# cars$Car<-ifelse(cars$Transport=="Car",1,0)
# cars$Public_Transport<-ifelse(cars$Transport=="Public Transport",1,0)

# str(cars)

## ------------------ split the data into train(70%) and test(30%) -----------------

library(caret)

# Set the seed for the purpose of reproducible results
set.seed(36756)

pd<-sample(2,nrow(cars),replace=TRUE, prob=c(0.7,0.3)) ### FOR dividing into 2 parts
train<-cars[pd==1,]
test<-cars[pd==2,]

c(nrow(train), nrow(test))
prop.table(table(train$Transport))
prop.table(table(test$Transport))


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


#--------- 2 variable KNN model for visualization -----------------------


train.2fact<-train[,c(1,6,9)]
val.2fact<-test[,c(1,6,9)]
str(train.2fact)
train.2fact$Transport<-as.factor(train.2fact$Transport)
val.2fact$Transport<-as.factor(val.2fact$Transport)

y_pred<-knn(train=train.2fact[,-3],test=val.2fact[-3], cl=train.2fact[,3],k=10)
cm.knn<-table(val.2fact[,3],y_pred)
cm.knn

accuracy.knn<-sum(diag(cm.knn))/sum(cm.knn)
accuracy.knn

# Visualizing Training set
set = train.2fact
X1 = seq(min(set[, 1])-0.1 , max(set[, 1])+0.1 , by = 0.01)
X2 = seq(min(set[, 2])-0.1, max(set[, 2])+0.1 , by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c( 'Work.Exp', 'Age')
y_grid = knn(train=train.2fact[,-3],test=grid_set[-3], cl=train.2fact[,3],k=20)
plot(set[, -3],
     main = 'Knn (Train set)',
     xlab = 'Work.Exp' , ylab = 'Age',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == "1", 'blue', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == "1", 'green', 'black'))


# Now for test
set = val.2fact
X1 = seq(min(set[, 1]) , max(set[, 1]) , by = 0.01)
X2 = seq(min(set[, 2]), max(set[, 2]) , by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c( 'Work.Exp', 'Age')
y_grid = knn(train=train.2fact[,-3],test=grid_set[-3], cl=train.2fact[,3],k=10)
plot(set[, -3],
     main = 'Knn (Val set)',
     xlab = 'Work.Exp' , ylab ='Age' ,
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == "1", 'blue', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == "1", 'green', 'black'))


#---------------------------- KNN model  ------------------------------------

train_final <- train[,c(9,1,3:8,10)]
test_final <- test[,c(9,1,3:8,10)]

str(train_final)  
str(test_final)

library(class)

set.seed(678999)
car_pred<-knn(train=train_final[,-1],test=test_final[,-1], cl=train_final[,1],k=10)
tab.knn<-table(test_final[,1],car_pred)
tab.knn

accuracy.knn<-sum(diag(tab.knn))/sum(tab.knn)
accuracy.knn ### 81.69 with set.seed(678999) ; k =10



#--------- 2 variable Naive Bayes model for visualization -----------------------

str(train)
train.2fact<-train[,c(1,8,9)]
val.2fact<-test[,c(1,8,9)]
str(train.2fact)
train.2fact$Transport<-as.factor(train.2fact$Transport)
val.2fact$Transport<-as.factor(val.2fact$Transport)


NB.1<-naiveBayes(x=train.2fact[-3], y=train.2fact$Transport)
y_pred<-predict(NB.1,newdata=val.2fact[-3])

#Confusion matrix
cm.NB.1=table(val.2fact[,3],y_pred)
cm.NB.1

# Visualising the Training set results

library(ElemStatLearn)
set = train.2fact
X1 = seq(min(set[, 1])-0.1 , max(set[, 1])+0.1 , by = 0.01)
X2 = seq(min(set[, 2]) -0.1, max(set[, 2])+0.1 , by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'license')
y_grid = predict(NB.1, newdata = grid_set)
plot(set[, -3],
     main = 'Naive Bayes (Train set)',
     xlab = 'Age', ylab = 'license',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == "1", 'blue', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == "1", 'green', 'black'))


#Now for test
library(ElemStatLearn)
set = val.2fact
X1 = seq(min(set[, 1])-0.1 , max(set[, 1])+0.1 , by = 0.01)
X2 = seq(min(set[, 2]) -0.1, max(set[, 2])+0.1 , by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'license')
y_grid = predict(NB.1, newdata = grid_set)
plot(set[, -3],
     main = 'Naive Bayes (Val set)',
     xlab = 'Age', ylab = 'license',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == "1", 'blue', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == "1", 'green', 'black'))



#--------------------------- Naive Bayes model ---------------------------


train_final <- train[,c(9,1,3:8,10)]
test_final <- test[,c(9,1,3:8,10)]

library(e1071)
NB<-naiveBayes(x=train_final[,-1], y=train_final$Transport)
y_pred<-predict(NB,newdata=test_final[-1])
str(y_pred)

#Confusion matrix
cm.NB=table(test_final[,1],y_pred)
cm.NB
accuracy.NB<-sum(diag(cm.NB))/sum(cm.NB)
accuracy.NB   ###  76.05


#----- --------------- Prediction for 2 records KNN -----

##setwd("C:\\mydata\\files")

car_test<-read.csv("cars_test_data.csv",header=TRUE)

str(car_test)

car_test$Female<-ifelse(car_test$Gender=="Female",1,0)

# 
# normalize<-function(x){
#   +return((x-min(x))/(max(x)-min(x)))}
# 
# car_test<-transform(car_test, Age=ave(car_test$Age,FUN = normalize))
# car_test<-transform(car_test, Work.Exp=ave(car_test$Work.Exp,FUN = normalize))
# car_test<-transform(car_test, Salary=ave(car_test$Salary,FUN = normalize))
# car_test<-transform(car_test, Distance=ave(car_test$Distance,FUN = normalize))


str(car_test)
car_test_final <- car_test[,c(1,3:9)]
str(car_test_final)

#----------------------
set.seed(678999)
car_pred1<-knn(train=train_final[,-1],test=car_test_final, cl=train_final[,1],k=10)
car_pred1

#----------------------------

car_test_final$PredictedTransport <- predict(NB, car_test_final)
car_test_final






























