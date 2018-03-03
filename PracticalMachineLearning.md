---
title: "My brief analysis about the Weight Lifting Exercises"
author: "Mauricio Paletta"
date: "March 03, 2018"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Synopsis

This report describes an own prediction model related to the Weight Lifting Exercises. The original work, including the data, is title "Qualitative Activity Recognition of Weight Lifting Exercises" and is from: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. (Proceedings of 4th International Conference in Cooperation with SIGCHI, Stuttgart, Germany, ACM SIGCHI, 2013).

In order to reproduce the same results presented in this paper, a seed equal to 965 was used.
 
## The data and cross validation strategy

The Weight Lifting Exercises consists of the following: Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions (the predictor): 

- Class A: Exactly according to the specification, 
- Class B: Throwing the elbows to the front, 
- Class C: Lifting the dumbbell only halfway, 
- Class D: Lowering the dumbbell only halfway, and 
- Class E: Throwing the hips to the front.

```{r, echo = FALSE, warning = FALSE, message = FALSE}
require(caret)
require(randomForest)
require(parallel)
require(doParallel)
require(naivebayes)

rawData <- read.csv("pml-training.csv")
```

The raw data has `r paste(nrow(rawData))` observations and `r paste(ncol(rawData))` variables. However several variables as for example those with the prefix "var_", "avg_", "stddev_" and others are unuseful for the machine learning model (these data have NA values). There are also several rows that can be removed from the raw data because are statistical results of capture windows (they are identified because they have the value "yes" in the variable "new_window").

On the other hand, the raw data is ordered according to the class (first the data classified as A, then the Bs and so on). For the training process is advisable to randomly regroup the data.

```{r, echo = FALSE}
# The seed to reproduce the same results
#
set.seed(965)

# Loading the data
#
rawData <- rawData[-which(rawData$new_window == "yes"), ]

# Filtering the data
#
rawData <- rawData[, c(8:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160)]

# Randomly regroup the raw data
#
rawData <- rawData[sample(nrow(rawData), nrow(rawData)), ]

# Training & testing sets
#
inTrain <- createDataPartition(y = rawData$classe, p = 0.75, list = FALSE)

training <- rawData[inTrain,]
testing <- rawData[-inTrain,]
```

After pre-processing the raw data, the resulting dataset has `r paste(nrow(rawData))` observations and `r paste(ncol(rawData))` variables. This dataset was randomly splitted in a training set of 75% of the data and a remaining 25% testing set.   

## The Model

After reviewing the remaining variables of the dataset and the original paper of this work, it is possible to notice 4 different groups of variables grouped according to the device for capturing the data: belt, arm, dumbbell and forearm.

In order to reduce the complexity of the problem (because there are many variables involved) and therefore reduce the calculation time for the training process, the strategy of this machine learning model is to make a model for each of these 4 groups in order to learn the influence of the capture device on the predictor and then a final model that learns from the combination of the 4 devices.

To decide which method to use in these models, let's first take a look at the variances of each of the variables and their influence on each of the 5 classes of the predictor:

```{r, echo = FALSE}
# Find the variances for any of the capture devices: belt, arm, dumbbell and forearm
#
classA <- subset(training, classe == "A")
classB <- subset(training, classe == "B")
classC <- subset(training, classe == "C")
classD <- subset(training, classe == "D")
classE <- subset(training, classe == "E")

dt <- data.frame(
    varA = apply(classA[, 1:52], 2, var),
    varB = apply(classB[, 1:52], 2, var),
    varC = apply(classC[, 1:52], 2, var),
    varD = apply(classD[, 1:52], 2, var),
    varE = apply(classE[, 1:52], 2, var)
    )
format(round(dt, 2), scientific = FALSE, nsmall = 2)
```

As can be seen in the table of variances, most of the variables have very high values, indicating that there is a lot of noise in the capture of the data. Then, following the same decision of the authors of this work, it is better to apply the random forest method for each of the models associated to the capture devices.

Aiming to reduce the complexity of the problem, instead of taking the three variables referred to a coordinate (x, y, z) the average of the three values is assumed. It is thus reduced to 1 variable for each group of three coordinates.

```{r, echo = FALSE}
# Parallel processing for improved performance of training 
#
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

aux <- training
aux <- cbind(aux, avgGyros_belt = apply(training[, 5:7], 1, mean))
aux <- cbind(aux, avgAccel_belt = apply(training[, 8:10], 1, mean))
aux <- cbind(aux, avgMagnet_belt = apply(training[, 11:13], 1, mean))
aux <- cbind(aux, avgGyros_arm = apply(training[, 18:20], 1, mean))
aux <- cbind(aux, avgAccel_arm = apply(training[, 21:23], 1, mean))
aux <- cbind(aux, avgMagnet_arm = apply(training[, 24:26], 1, mean))
aux <- cbind(aux, avgGyros_dumbbell = apply(training[, 31:33], 1, mean))
aux <- cbind(aux, avgAccel_dumbbell = apply(training[, 34:36], 1, mean))
aux <- cbind(aux, avgMagnet_dumbbell = apply(training[, 37:39], 1, mean))
aux <- cbind(aux, avgGyros_forearm = apply(training[, 44:46], 1, mean))
aux <- cbind(aux, avgAccel_forearm = apply(training[, 47:49], 1, mean))
aux <- cbind(aux, avgMagnet_forearm = apply(training[, 50:52], 1, mean))
training <- aux

# Fit the models for any of the capture devices: belt, arm, dumbbell and forearm
#
modBelt <- randomForest(classe ~ roll_belt + pitch_belt + yaw_belt + 
                            total_accel_belt + avgGyros_belt + avgAccel_belt +
                            avgMagnet_belt, 
                        data = training)
modArm <- randomForest(classe ~ roll_arm + pitch_arm + yaw_arm + total_accel_arm +
                           avgGyros_arm + avgAccel_arm + avgMagnet_arm,
                       data = training)
modDumbbell <- randomForest(classe ~ roll_dumbbell + pitch_dumbbell + yaw_dumbbell +
                                total_accel_dumbbell + avgGyros_dumbbell +
                                avgAccel_dumbbell + avgMagnet_dumbbell,
                            data = training)
modForearm <- randomForest(classe ~ roll_forearm + pitch_forearm + yaw_forearm +
                               total_accel_forearm + avgGyros_forearm +
                               avgAccel_forearm + avgMagnet_forearm,
                           data = training)

aux <- testing
aux <- cbind(aux, avgGyros_belt = apply(testing[, 5:7], 1, mean))
aux <- cbind(aux, avgAccel_belt = apply(testing[, 8:10], 1, mean))
aux <- cbind(aux, avgMagnet_belt = apply(testing[, 11:13], 1, mean))
aux <- cbind(aux, avgGyros_arm = apply(testing[, 18:20], 1, mean))
aux <- cbind(aux, avgAccel_arm = apply(testing[, 21:23], 1, mean))
aux <- cbind(aux, avgMagnet_arm = apply(testing[, 24:26], 1, mean))
aux <- cbind(aux, avgGyros_dumbbell = apply(testing[, 31:33], 1, mean))
aux <- cbind(aux, avgAccel_dumbbell = apply(testing[, 34:36], 1, mean))
aux <- cbind(aux, avgMagnet_dumbbell = apply(testing[, 37:39], 1, mean))
aux <- cbind(aux, avgGyros_forearm = apply(testing[, 44:46], 1, mean))
aux <- cbind(aux, avgAccel_forearm = apply(testing[, 47:49], 1, mean))
aux <- cbind(aux, avgMagnet_forearm = apply(testing[, 50:52], 1, mean))
testing <- aux

predBelt <- predict(modBelt, testing)
predArm <- predict(modArm, testing)
predDumbbell <- predict(modDumbbell, testing)
predForearm <- predict(modForearm, testing)

predDF <- data.frame(predBelt, predArm, predDumbbell, predForearm, 
                     realClass = testing$classe)
```

Plot 1 shows the first 100 results of the predictions of the models of the 4 devices. The upper line shows the hits between the predicted and the real class; the bottom line shows the inequalities. As we can seen, although there are a lot of successes in each case, there are still many inequalities (the accuracy is `r paste(sprintf("%.2f", mean(length(which(predBelt == testing$classe)), length(which(predArm == testing$classe)), length(which(predDumbbell == testing$classe)), length(which(predForearm == testing$classe))) / length(testing$classe)))`). 

```{r, echo = FALSE}
pBelt <- vector(length = length(testing$classe))
pArm <- vector(length = length(testing$classe))
pDumbbell <- vector(length = length(testing$classe))
pForearm <- vector(length = length(testing$classe))
for (i in 1:length(testing$classe)) {
  pBelt[i] <- if (predBelt[i] == testing$classe[i]) 1 else 0.5
  pArm[i] <- if (predArm[i] == testing$classe[i]) 2 else 1.5
  pDumbbell[i] <- if (predDumbbell[i] == testing$classe[i]) 3 else 2.5
  pForearm[i] <- if (predForearm[i] == testing$classe[i]) 4 else 3.5
}
pDF <- data.frame(Belt = pBelt, Arm = pArm, Dumbbell = pDumbbell, Forearm = pForearm)

g <- ggplot(pDF[1:100, ], aes(1:100)) + 
  geom_point(aes(y = pBelt[1:100]), colour = "red3") + 
  geom_point(aes(y = pArm[1:100]), colour = "green3") +
  geom_point(aes(y = pDumbbell[1:100]), colour = "orchid3") +
  geom_point(aes(y = pForearm[1:100]), colour = "cyan3") +
  labs(title = "Plot 1. Real vs predicted values for capture devices models") +
  labs(x = "Index") +
  scale_y_discrete(name = element_blank(), 
                   limits = c("1", "2", "3", "4"),
                   labels = c("1" = "Belt", "2" = "Arm", "3" = "Dumbbell", "4" = "Forearm"))
g

```

The next and final step is combine the 4 predictors in a final general model. For this case I chose the method Naive Bayes because it assumes independence between features as well as that each of the class densities are products of marginal densities (they assume that the inputs are conditionally independent in each class). The confusion matrix and accuracy is as follows:

```{r, echo = FALSE}
# Fit the general model and make the predictions
#
combMod <- naive_bayes(realClass ~ ., data = predDF)

combPred <- predict(combMod, predDF)

# Calculate the confusion matrix
#
cm <- confusionMatrix(combPred, predDF$realClass)
cm

# Stop parallel processing
#
stopCluster(cluster)
registerDoSEQ()
```

Plot 2 shows the comparison between the real and prediction values resulting from the general model. As we can seen the accuracy is `r paste(sprintf("%.2f", cm$overall[1]))`.

```{r, echo = FALSE}
p <- vector(length = length(predDF$realClass))
for (i in 1:length(predDF$realClass)) {
  p[i] <- if (combPred[i] == predDF$realClass[i]) 2 else 1
}
pDF <- data.frame(Predicted = p, realClass = predDF$realClass)

g <- ggplot(pDF, aes(1:nrow(pDF))) + 
  geom_point(aes(y = p)) + 
  labs(title = "Plot 2. Real vs predicted values for the general model") +
  labs(x = "Index") +
  scale_y_discrete(name = "Predicted classes", 
                   limits = c("1", "2"),
                   labels = c("1" = "No", "2" = "OK"))
g

```

## Using the original test file

Below the results (classification) by using the model presented in this paper with the test file given by the authors of this work.

```{r, echo = FALSE}
# Loading and preparing the data
#
rawData <- read.csv("pml-testing.csv")
rawData <- rawData[, c(8:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160)]

aux <- rawData
aux <- cbind(aux, avgGyros_belt = apply(rawData[, 5:7], 1, mean))
aux <- cbind(aux, avgAccel_belt = apply(rawData[, 8:10], 1, mean))
aux <- cbind(aux, avgMagnet_belt = apply(rawData[, 11:13], 1, mean))
aux <- cbind(aux, avgGyros_arm = apply(rawData[, 18:20], 1, mean))
aux <- cbind(aux, avgAccel_arm = apply(rawData[, 21:23], 1, mean))
aux <- cbind(aux, avgMagnet_arm = apply(rawData[, 24:26], 1, mean))
aux <- cbind(aux, avgGyros_dumbbell = apply(rawData[, 31:33], 1, mean))
aux <- cbind(aux, avgAccel_dumbbell = apply(rawData[, 34:36], 1, mean))
aux <- cbind(aux, avgMagnet_dumbbell = apply(rawData[, 37:39], 1, mean))
aux <- cbind(aux, avgGyros_forearm = apply(rawData[, 44:46], 1, mean))
aux <- cbind(aux, avgAccel_forearm = apply(rawData[, 47:49], 1, mean))
aux <- cbind(aux, avgMagnet_forearm = apply(rawData[, 50:52], 1, mean))
testing <- aux

# Run the model
#
predBelt <- predict(modBelt, testing)
predArm <- predict(modArm, testing)
predDumbbell <- predict(modDumbbell, testing)
predForearm <- predict(modForearm, testing)
predDF <- data.frame(predBelt, predArm, predDumbbell, predForearm)
predict(combMod, predDF)

```

## Summary

It was important to carry out a preliminary analysis of the data. The application of filters allowed to reduce the complexity of the learning process. An analysis of the variances of the variables allowed to decide the method to be used in the model. The use of the strategy of combination of models allowed to obtain good results in the prediction.
