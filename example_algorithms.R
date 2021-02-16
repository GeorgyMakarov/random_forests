library(tidyverse)
library(dplyr)
library(dslabs)
library(ggplot2)
library(randomForest)
library(caret)

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