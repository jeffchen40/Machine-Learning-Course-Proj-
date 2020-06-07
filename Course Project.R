#Loading Libraries and Reading Data
library(caret) ; library(rmarkdown) ; library(ggplot2) ; library(randomForest)
train <- read.csv('pml-training.csv', header = T)
valid <- read.csv('pml-testing.csv', header = T)
names(train)
train$classe
unique(train$classe)

#Cleaning Data - removing NA values from columns
train <- train[, colSums(is.na(train))==0]
valid <- valid[, colSums(is.na(valid))==0]

#Cleaning Data - removing first 7 variables as they have little impact on 'classe'
train <- train[, -c(1:7)]
valid <- valid[, -c(1:7)]

#Preparing data for prediction - splitting into 70% training, 30% testing
set.seed(1234)
inTrain <- createDataPartition(y = train$classe,
                               p = 0.7, list = FALSE)
training <- train[inTrain, ]
testing <- train[-inTrain, ]
dim(training)
dim(testing)

#Turning classe to a factor variable
training$classe <- factor(training$classe)
testing$classe <- factor(testing$classe)

#Further cleaning to remove variables with near-zero-variance
nzv <- nearZeroVar(training)
training <- training[,-nzv]
testing  <- testing[,-nzv]
dim(training)
dim(testing)

#Predicting with Classification Tree
set.seed(12345)
mod_rpart <- train(classe ~., method = 'rpart', data = training)
library(rattle)
fancyRpartPlot(mod_rpart$finalModel) ##using Rattle to plot classification tree
pred_rpart <- predict(mod_rpart, newdata = testing)
cm_rpart <- confusionMatrix(pred_rpart, testing$classe)
cm_rpart
cm_rpart$overall['Accuracy']
##Classification Tree Accuracy: 0.4937978 which is low; out of sample error ~0.5

#Predicting with Random Forest
set.seed(12345)
control_rf <- trainControl(method = 'cv', number = 3, verboseIter = FALSE)
mod_rf <- randomForest(classe ~., data = training, trControl = control_rf)
pred_rf <- predict(mod_rf, testing)
cm_rf <- confusionMatrix(pred_rf, testing$classe)
cm_rf
cm_rf$overall['Accuracy']
##Random Forest Accuracy: 0.9959218; out of sample error near 0 but may be due to overfitting
plot(mod_rf)

#Therefore, by looking at the accuracy rates, Random Forest is the best model for prediction
#Using Random Forest on validation data
results <- predict(mod_rf, newdata = valid)
results

