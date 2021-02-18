library(tidyverse)
library(dplyr)
library(dslabs)
library(ggplot2)
library(randomForest)
library(caret)
library(rpart)

# Decision tree motivation example
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#classification-and-regression-trees-cart
data("olive")
names(olive)
table(olive$region)

olive = select(olive, -area)
fit   = train(region ~.,
              method   = "knn",
              tuneGrid = data.frame(k = seq(1, 15, 2)),
              data     = olive)
ggplot(fit)

olive %>% gather(fatty_acid, percentage, -region) %>%
    ggplot(aes(region, percentage, fill = region)) +
    geom_boxplot() +
    facet_wrap(~fatty_acid, scales = "free", ncol = 4) +
    theme(axis.text.x = element_blank(), legend.position="bottom")


olive %>% 
    ggplot(aes(eicosenoic, linoleic, color = region)) + 
    geom_point() +
    geom_vline(xintercept = 0.065, lty = 2) + 
    geom_segment(x = -0.2, y = 10.54, xend = 0.065, yend = 10.54, 
                 color = "black", lty = 2)
rm(list = ls())


# Regression tree example
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#cart-motivation
data("polls_2008")
qplot(day, margin, data = polls_2008)
summary(polls_2008$day)


library(rpart)
fit = rpart(margin ~ ., data = polls_2008)
plot(fit, margin = 0.1)
text(fit, cex = 0.75)

polls_2008 %>% 
    mutate(y_hat = predict(fit)) %>% 
    ggplot() +
    geom_point(aes(day, margin)) +
    geom_step(aes(day, y_hat), col="red")


fit <- rpart(margin ~ ., 
             data = polls_2008, 
             control = rpart.control(cp = 0, minsplit = 2))
polls_2008 %>% 
    mutate(y_hat = predict(fit)) %>% 
    ggplot() +
    geom_point(aes(day, margin)) +
    geom_step(aes(day, y_hat), col="red")


# Example of choosing parameters
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#cart-motivation
train_rpart = train(margin ~.,
                    method   = "rpart",
                    tuneGrid = data.frame(cp = seq(0.00, 0.05, len = 25)),
                    data     = polls_2008)
ggplot(train_rpart)
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)

polls_2008 %>% 
    mutate(y_hat = predict(train_rpart)) %>% 
    ggplot() +
    geom_point(aes(day, margin)) +
    geom_step(aes(day, y_hat), col="red")


pruned_fit = prune(fit, cp = 0.01)
polls_2008 %>% 
    mutate(y_hat = predict(pruned_fit)) %>% 
    ggplot() +
    geom_point(aes(day, margin)) +
    geom_step(aes(day, y_hat), col="red")
rm(list = ls())

# Example of classification decision tree
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#cart-motivation
train_rpart <- train(y ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = mnist_27$train)
plot(train_rpart)

y_hat <- predict(train_rpart, mnist_27$test)
confusionMatrix(y_hat, mnist_27$test$y)$overall["Accuracy"]
rm(list = ls())


# Example of random forest
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#random-forests
data(polls_2008)

plot(x     = polls_2008$day,
     y     = polls_2008$margin,
     col   = "dodgerblue1",
     pch   = 19,
     main  = "Explore polls data",
     frame = F,
     xlab  = "day",
     ylab  = "margin")
points(x     = polls_2008$day,
       y     = polls_2008$margin,
       pch   = 21)

fit = randomForest(margin ~., data = polls_2008)
plot(fit,
     frame = F,
     col   = "blue")

polls_2008 %>% 
    mutate(y_hat = predict(fit, newdata = polls_2008)) %>% 
    ggplot() +
    geom_point(aes(day, margin), col = "dodgerblue1") +
    geom_line(aes(day, y_hat), col = "red") +
    theme_bw()

rm(list = ls())


# Digits example
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#random-forests
mydata = mnist_27$train
head(mydata)

train_rf = randomForest(y ~., data = mnist_27$train)
caret::confusionMatrix(predict(train_rf, mnist_27$test),
                       mnist_27$test$y)$overall["Accuracy"]


# Optimize node size
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#random-forests
library(caret)
nodesize = seq(1, 51, 10)
acc = sapply(nodesize, function(ns){train(y ~.,
                                          method   = "rf",
                                          data     = mnist_27$train,
                                          tuneGrid = data.frame(mtry = 2),
                                          nodesize = ns)$results$Accuracy})

qplot(nodesize, acc)

train_rf_2 = randomForest(y ~., 
                          data = mnist_27$train,
                          nodesize = nodesize[which.max(acc)])
caret::confusionMatrix(predict(train_rf_2, mnist_27$test),
                       mnist_27$test$y)$overall["Accuracy"]
rm(list = ls())

# Show importance of variables
# Source: https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/importance
set.seed(4543)
data(mtcars)
mtcars_rf = randomForest(mpg ~.,
                         data        = mtcars,
                         ntree       = 1000,
                         keep.forest = FALSE,
                         importance  = TRUE)
importance(mtcars_rf)
importance(mtcars_rf, type = 1)


# Exercises:part 1
# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html

library(caret)
library(dplyr)

# 1. Create a simple dataset where the outcome grows 0.75 units on average for 
#    every increase in a predictor
n     = 1e3
sigma = 0.25
set.seed(123)
x     = rnorm(n, 0, 1)
y     = 0.75 * x + rnorm(n, 0, sigma)
data  = data.frame(x = x, y = y)
rm(n, x, y, sigma)

# 2. Use rpart to create a regression tree and save results to `fit`
fit_rpart = rpart::rpart(y ~ x, data = data)
plot(fit_rpart, margin = 0.1)
text(fit_rpart, cex = 0.75)

# 3. Make a scatterplot of y vs x values along with predicted values based on `fit`
data %>% 
    mutate(y_hat = predict(fit_rpart)) %>% 
    ggplot() +
    geom_point(aes(x, y), col = "darkgrey") +
    geom_step(aes(x, y_hat), col="red", lwd = 1.1)

# 4. Model random forest using `randomForest` package
fit_rf1 = randomForest(y ~ x, data = data)
data %>% 
    mutate(y_hat = predict(fit_rf1, newdata = data)) %>% 
    ggplot() +
    geom_point(aes(x, y), col = "darkgrey") +
    geom_line(aes(x, y_hat), col = "red")

# 5. Check if random forest error is stable
plot(fit_rf1, col = "blue")

# 6. Set nodesize to 50 and maxnodes to 25. Compare the smootheness of two fits.
fit_rf2 = randomForest(y ~ x,
                       data     = data,
                       nodesize = 50,
                       maxnodes = 25)

data %>% 
    mutate(y_hat1 = predict(fit_rf1, newdata = data),
           y_hat2 = predict(fit_rf2, newdata = data)) %>% 
    ggplot() +
    geom_point(aes(x, y), col = "darkgrey") +
    geom_line(aes(x, y_hat1), col = "red") +
    geom_line(aes(x, y_hat2), col = "blue")

# 7. Use `train` function to help pick up correct values
mn        = seq(5, 250, 25)
rmse_logs = sapply(mn, function(mn){
    caret::train(y ~ x,
                 method = "Rborist",
                 data   = mydata,
                 tuneGrid = data.frame(predFixed = 1,
                                       minNode   = 20))$results$RMSE
})
qplot(mn, rmse_logs)

train_rf_3 = randomForest(y ~ x,
                          data = data,
                          nodesize = mn[which.min(rmse_logs)])

# 8. Make scatter plot with predictions from best model
data %>% 
    mutate(y_hat1 = predict(fit_rf1, newdata = data),
           y_hat2 = predict(fit_rf2, newdata = data),
           y_hat3 = predict(train_rf_3, newdata = data)) %>% 
    ggplot() +
    geom_point(aes(x, y), col = "darkgrey") +
    geom_line(aes(x, y_hat1), col = "dodgerblue1") +
    geom_line(aes(x, y_hat2), col = "blue") +
    geom_line(aes(x, y_hat3), col = "red")


# 9. Use `rpart` function to fit a classification tree to the dataset
#    `tissue_gene_expression`
mydata = tissue_gene_expression
rm(tissue_gene_expression)

rpart_1 = rpart(y ~ x, data = mydata)
summary(rpart_1)
rpart_2 = with(mydata, train(x, y,
                             method = "rpart",
                             tuneGrid = data.frame(cp = seq(0.00, 0.05, 0.01))))
ggplot(rpart_2)

# 10. Study the confusion matrix
y_hat = predict(rpart_2, newdata = mydata$x)
confusionMatrix(y_hat, mydata$y)$overall["Accuracy"]

# 11. Does the change in minnode increase the accuracy?

# 12. Plot the tree from the best fitting model
plot(rpart_2$finalModel, margin = 0.1)
text(rpart_2$finalModel, cex = 0.75)
rpart_2$finalModel

# 13. Grow a random forest with nodesize = 1, mtry = seq(50, 200, 25)
set.seed(1990)
train_rf_1 = with(mydata,
                  train(x, y,
                        method   = "rf",
                        tuneGrid = data.frame(mtry = seq(50, 200, 25)),
                        nodesize = 1))
acc  = train_rf_1$results$Accuracy
mtry = train_rf_1$results$mtry
qplot(mtry, acc)

# 14. Extract variables importance
var_imp = train_rf_1$finalModel$importance
var_imp = data.frame(var_imp)
var_imp = var_imp %>% arrange(desc(MeanDecreaseGini))

rm(list = ls())