# Exercises: part 2
# Source: self created

library(dplyr)
library(ggplot2)
library(caret)


# Regression
# Outcome decreases 1 / x ^ 2 on every increase in a predictor
n = 1e3
s = 0.25
set.seed(123)
x = rnorm(n, 2, 0.5)
y = (1 / (x ^ 2)) + rnorm(n, 0, s)
d = data.frame(x, y)
rm(n, s, x, y)

# Grow default random forest using caret
rf_def = train(y ~ x,
               data   = d,
               method = "rf")

# Tune parameters of random forest
rf_tun = train(y ~ x,
               data     = d,
               method   = "Rborist",
               tuneGrid = data.frame(predFixed = 1,
                                     minNode   = seq(5, 250, 25)))

# Extract the nodesize with minimum rmse
res = 
    rf_tun$results %>% 
    select(minNode, RMSE) %>% 
    filter(RMSE == min(RMSE))
min_rmse = res$minNode
rm(res)

# Plot the results
d %>% 
    mutate(y_hat_def = predict(rf_def, newdata = d),
           y_hat_tun = predict(rf_tun, newdata = d)) %>% 
    ggplot() +
    geom_point(aes(x, y),        col = "lightgrey") +
    geom_line(aes(x, y_hat_def), col = "darkgrey") +
    geom_line(aes(x, y_hat_tun), col = "red")
