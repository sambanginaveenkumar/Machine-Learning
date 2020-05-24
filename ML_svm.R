# Saving data in time series format #

#Install required Packages

# install.packages("ggplot2")
# install.packages("car")
# install.packages("scales")
# install.packages("AER")
# install.packages("tidyr")
# install.packages("corrplot")
# install.packages("caret")
# install.packages("purrr")
# install.packages("coefplot")
# install.packages("psych")
# install.packages("MASS")
# install.packages("leaflet.extras")
# install.packages("PerformanceAnalytics")
# install.packages("VIM")
# install.packages("DMwR")
# install.packages("RxCEcolInf")
# install.packages("e1071")
# install.packages("ggcorrplot")
# install.packages("mlogit")
# install.packages("caTools")
# install.packages("RColorBrewer")
# install.packages("dummies")
# install.packages("pscl")
# install.packages("StatMeasures")
# install.packages("sqldf")
# install.packages("lubridate")
# install.packages("glmnet")

library(ggplot2)
library(car)
library(scales)
library(AER)
library(tidyr)
library(corrplot)
library(caret)
library(purrr)
library(coefplot)
library(psych)
library(MASS)
library(leaflet.extras)
library(PerformanceAnalytics)
library(VIM)
library(DMwR)
library(RxCEcolInf)
library(e1071)
library(ggcorrplot)
library(mlogit)
library(caTools)
library(RColorBrewer)
library(dummies)
library(pscl)
library(StatMeasures)
library(sqldf)
library(lubridate)
library(glmnet)

setwd("E:/r direct/Machine Learning/Assignment")
cars<-read.csv("Cars.csv",header=TRUE)

# dim(car)
# 
# summary(car)
# 
# str(car)
# 
# hist(car$Work.Exp, col = 'violet')
# hist(car$Age, col = 'violet')
# hist(car$Distance, col = 'violet')
# hist(car$license, col = 'violet')
# 
# car_summary <- ggplot(car, aes(x = car$Salary, y = car$Work.Exp)) +
#   facet_grid(~ car$Gender + car$Transport)+
#   geom_boxplot(na.rm = TRUE, colour = "#3366FF",outlier.colour = "red", outlier.shape = 1) +
#   labs(x = "Work Experience", y = "Salary") +
#   scale_x_continuous() +
#   scale_y_continuous() +
#   theme(legend.position="bottom", legend.direction="horizontal")
# car_summary


str(cars)
car_imputed = car

car_imputed <- VIM::kNN(data=car,variable =c("MBA"),k=20)
summary(car_imputed)

car_final <- subset(car_imputed, select = Age:Transport)
car_final_boost <- subset(car_imputed, select = Age:Transport)
car_final_logit <- subset(car_imputed, select = Age:Transport)

table(car_final$Transport)
print(prop.table(table(car_final$Transport)))


#----- data split into train (70%) and test (30%) ----

library(caret)

# Set the seed for the purpose of reproducible results
set.seed(786449)

inTrain <- createDataPartition(car_final$Transport,p=0.70,list = FALSE)
Training <- car_final[inTrain,]
Testing <- car_final[-inTrain,]

c(nrow(Training), nrow(Testing))
prop.table(table(Training$Transport))
prop.table(table(Testing$Transport))

str(Training)


# BUILD SUPPORT VECTOR MACHINE MODEL:

library(e1071)

#----------------------- Radial Kernal without tuning -----------------------

Radialsvm<-svm(Transport~., data=Training, kernel="radial")
summary(Radialsvm)


#----- Prediction on Train data--------------

pred_on_train <- predict(Radialsvm, Training)
confusionMatrix(Training$Transport,pred_on_train)  ##82.37


#----- Prediction on Test data --------------

pred_on_test <- predict(Radialsvm, Testing)
confusionMatrix(Testing$Transport,pred_on_test)   ##79.55



#---------------------------- Radial Kernal with tuning ------------

set.seed(4374) # 84, 81 ; seq(0,1,0.01), cost = 2^(-1:3)

# set.seed(90345) 85, 80 ; seq(0,1,0.01), cost = 2^(-2:2)

svm_tune <- tune(svm, Transport~., data = Training, 
                 ranges = list( epsilon = seq(0,1,0.01), cost = 2^(-1:3)))
print(svm_tune)
summary(svm_tune)

best_mod <- svm_tune$best.model
summary(best_mod)

best.par<-svm_tune$best.parameters
summary(best.par)

# --- prediction 

pred <- predict(best_mod, Training)
confusionMatrix(Training$Transport,pred)   ##84.29

prediction_on_test <- predict(best_mod, Testing)
confusionMatrix(Testing$Transport,prediction_on_test) ## 81.82


#---------------------- VARIABLE IMPORTANCE -----------------------------


weightV <- t(best_mod$coefs) %*% best_mod$SV          # weight vectors
wkk <- apply(weightV, 2, function(v){sqrt(sum(v^2))})  # weight
wkkFinal <- sort(wkk, decreasing = T)
print(wkkFinal)

#---- 2 variable model----------
Training$Female<-ifelse(Training$Gender=="Female",1,0)
Testing$Female<-ifelse(Testing$Gender=="Female",1,0)

normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}

Training<-transform(Training, Age=ave(Training$Age,FUN = normalize))
Training<-transform(Training, Work.Exp=ave(Training$Work.Exp,FUN = normalize))
Training<-transform(Training, Salary=ave(Training$Salary,FUN = normalize))
Training<-transform(Training, Distance=ave(Training$Distance,FUN = normalize))

Testing<-transform(Testing, Age=ave(Testing$Age,FUN = normalize))
Testing<-transform(Testing, Work.Exp=ave(Testing$Work.Exp,FUN = normalize))
Testing<-transform(Testing, Salary=ave(Testing$Salary,FUN = normalize))
Testing<-transform(Testing, Distance=ave(Testing$Distance,FUN = normalize))


str(Training)
str(Testing)
train.2fact<-Training[,c(5,6,9)]
val.2fact<-Testing[,c(5,6,9)]
str(train.2fact)
train.2fact$Transport<-as.factor(train.2fact$Transport)
val.2fact$Transport<-as.factor(val.2fact$Transport)

# y_pred<-knn(train=train.2fact[,-3],test=val.2fact[-3], cl=train.2fact[,3],k=10)
# cm.knn<-table(val.2fact[,3],y_pred)
# cm.knn
# 
# accuracy.knn<-sum(diag(cm.knn))/sum(cm.knn)
# accuracy.knn

# Visualizing Training set


set = train.2fact
X1 = seq(min(set[, 1])-0.1 , max(set[, 1])+0.1 , by = 0.01)
X2 = seq(min(set[, 2])-0.1, max(set[, 2])+0.1 , by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c( 'Work.Exp', 'Age')
y_grid =svm(Transport~., data=Training, kernel="radial")
##lapply(y_grid, as.numeric)
plot(set[, -3],
     main = 'SVM (Train set)',
     xlab = 'Work.Exp' , ylab = 'Age',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, outer(X1, sqrt(abs(X1)), FUN = "/"), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == "1", 'blue', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == "1", 'green', 'black'))


# Now for test
set = val.2fact
X1 = seq(min(set[, 1]) , max(set[, 1]) , by = 0.01)
X2 = seq(min(set[, 2]), max(set[, 2]) , by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c( 'Work.Exp', 'Age')
y_grid =svm(Transport~., data=Training, kernel="radial")
plot(set[, -3],
     main = 'SVM (Val set)',
     xlab = 'Work.Exp' , ylab ='Age' ,
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, outer(X1, sqrt(abs(X1)), FUN = "/"), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == "1", 'blue', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == "1", 'green', 'black'))










#-----------------------------------------------------------------------

car_final_boost$Transport <- ifelse(car_final_boost$Transport == "Car",1,0)
table(car_final_boost$Transport )

car_final_boost <- dummy.data.frame(car_final_boost, sep = ".")
summary(car_final_boost)

boxplot(car_final_boost$Age ~ car_final_boost$Transport)
boxplot(car_final_boost$Work.Exp ~ car_final_boost$Transport)
boxplot(car_final_boost$Distance ~ car_final_boost$Transport)
boxplot(car_final_boost$Salary ~ car_final_boost$Transport)
boxplot(car_final_boost$Gender.Female ~ car_final_boost$Transport)

ggplot(data=car_final_boost, aes(x= car_final_boost$Gender.Female)) + 
  geom_histogram(col="red",fill="blue", bins = 25) +
  facet_grid(~ car_final_boost$Transport)+
  theme_bw()
