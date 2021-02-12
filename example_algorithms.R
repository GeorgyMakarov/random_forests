# Source: https://rafalab.github.io/dsbook/examples-of-algorithms.html#random-forests
# Example of random forest
library(dplyr)
library(dslabs)
library(ggplot2)
library(randomForest)
data(polls_2008)

# Explore data
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


# Fit random forest model
fit = randomForest(margin ~., data = polls_2008)
plot(fit,
     frame = F,
     col   = "blue")


# Make fit plot
polls_2008 %>% 
    mutate(y_hat = predict(fit, newdata = polls_2008)) %>% 
    ggplot() +
    geom_point(aes(day, margin), col = "dodgerblue1") +
    geom_line(aes(day, y_hat), col = "red") +
    theme_bw()

rm(list = ls())

# Digits example
mydata = mnist_27$train
head(mydata)

train_rf = randomForest(y ~., data = mnist_27$train)
caret::confusionMatrix(predict(train_rf, mnist_27$test),
                       mnist_27$test$y)$overall["Accuracy"]
