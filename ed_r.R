rm(list=ls(all=T))
setwd("E:/Subject/edwisor/project 2")
library(ggplot2)
library(gridExtra)
library(dplyr)
library(corrplot)
library(pROC)
library(C50)
library(caret)
library(rpart)
train = read.csv('Train_data.csv')
test = read.csv('Test_data.csv')

#Exploratory data analysis

var1 =ggplot(train, aes(area.code, fill = Churn)) + geom_bar(position = "fill") + labs(x = "Area code", y = "") + theme(legend.position = "none")
var2 =ggplot(train, aes(international.plan, fill = Churn)) + geom_bar(position = "fill") + labs(x = "International?", y = "") + theme(legend.position = "none")
var3 = ggplot(train, aes(voice.mail.plan, fill = Churn)) + geom_bar(position = "fill") + labs(x = "Voicemail?", y = "") + theme(legend.position = "none") 
var4 = ggplot(train, aes(number.customer.service.calls, fill = Churn)) + geom_bar(position = "fill") + labs(x = "Customer calls", y = "") + theme(legend.position = "none")


grid.arrange(var1, var2, var3, var4, ncol = 4, nrow = 1, top = "Churn & Non-Churn Chart")

#Explore distributions by continuous predictors

daymin = ggplot(train, aes(Churn, total.day.minutes, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
evemin = ggplot(train, aes(Churn, total.eve.minutes, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
nitmin = ggplot(train, aes(Churn, total.night.minutes, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
intmin =- ggplot(train, aes(Churn, total.intl.minutes, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
daycal = ggplot(train, aes(Churn, total.day.calls, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
evecal = ggplot(train, aes(Churn, total.eve.calls, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
nitcal = ggplot(train, aes(Churn, total.night.calls, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
intcal = ggplot(train, aes(Churn, total.intl.calls, fill = Churn)) + geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
grid.arrange(daymin, evemin, nitmin, intmin, 
             daycal, evecal, nitcal, intcal, 
             ncol = 4, nrow = 2)

#Find Missing values
anyNA(train)

#Check for collinearity.

corrplot(cor(train[sapply(train, is.numeric)]))

#Remove unnecessary features, which not required for train
train$state = NULL
train$area.code = NULL
train$phone.number = NULL
train$international.plan = NULL
train$voice.mail.plan = NULL

test$state = NULL
test$area.code = NULL
test$phone.number = NULL
test$international.plan = NULL
test$voice.mail.plan = NULL

#Encode dependent variable 
train$Churn = factor(train$Churn,
                           levels = c(' False.', ' True.'),
                           labels = c(0, 1))


test$Churn = factor(test$Churn,
                     levels = c(' False.', ' True.'),
                     labels = c(0, 1))

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = train[-1],
                          y = train$Churn,
                          ntree = 500)
y_pred = predict(classifier, newdata = test[-1])

cm = table(test[, 16], y_pred)

# Fitting Naive Bayes Classification to the Training set
#install.packages('e1071')
library(e1071)
classifier_n = naiveBayes(x = train[-1],
                        y = train$Churn)

# Predicting the Test set results
y_pred_n = predict(classifier_n, newdata = test[-1])


# Making the Confusion Matrix
cm_n = table(test[, 16], y_pred_n)



