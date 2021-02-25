# Explore random forest tuning based on Ames Housing example

# Dependencies
library(dplyr)
library(rsample)
library(randomForest)
library(ranger)
library(caret)


# Create training and testing sets from Ames Housing
set.seed(123)
ames_split = initial_split(AmesHousing::make_ames(), prop = 0.7)
ames_train = training(ames_split)
ames_test  = testing(ames_split)


# Grow default random forest
set.seed(123)
m1 = randomForest(formula = Sale_Price ~.,
                  data    = ames_train)
plot(m1, col = "blue", frame = F)
m1


# Find which number of trees provides the lowest error rate
# Compute average home sales price error
which.min(m1$mse)
sqrt(m1$mse[which.min(m1$mse)])
m1$rsq[which.max(m1$rsq)]


# Make part of training set a validation set to measure the accuracy without
# OOB samples
set.seed(123)
valid_split   = initial_split(ames_train, p = 0.8)
ames_train_v2 = analysis(valid_split)
ames_valid    = assessment(valid_split)
x_test        = ames_valid[setdiff(names(ames_valid), "Sale_Price")]
y_test        = ames_valid$Sale_Price

rf_oob_comp   = randomForest(formula = Sale_Price ~.,
                             data    = ames_train_v2,
                             xtest   = x_test,
                             ytest   = y_test)


# Extract OOB & validation errors
oob        = sqrt(rf_oob_comp$mse)
validation = sqrt(rf_oob_comp$test$mse)
tibble::tibble(`Out of Bag` = oob,
               `Test error` = validation,
               ntrees       = 1:rf_oob_comp$ntree) %>% 
    gather(Metric, RMSE, -ntrees) %>% 
    ggplot(aes(ntrees, RMSE, color = Metric)) +
    geom_line() +
    scale_y_continuous(labels = scales::dollar) +
    xlab("Number of trees")


# Initial tuning with randomForest
features = setdiff(names(ames_train), "Sale_Price")
set.seed(123)
m2 = tuneRF(x         = ames_train[features],
            y         = ames_train$Sale_Price,
            ntreeTry  = 500,
            mtryStart = 5,
            improve   = 0.01,
            trace     = F)

# Full grid search with ranger
## Construct grid of hyperparameters
hyper_grid = expand.grid(mtry        = seq(20, 30, by = 2),
                         node_size   = seq(3, 9, by = 2),
                         sample_size = c(0.55, 0.632, 0.70, 0.80),
                         OOB_RMSE    = 0)

## Run ranger models for each row in hyper_grid
for (i in 1:nrow(hyper_grid)){
    model = ranger(formula = Sale_Price ~.,
                   data            = ames_train,
                   num.trees       = 500,
                   mtry            = hyper_grid$mtry[i],
                   min.node.size   = hyper_grid$node_size[i],
                   sample.fraction = hyper_grid$sample_size[i],
                   seed            = 123)
    hyper_grid$OOB_RMSE[i] = sqrt(model$prediction.error)
}

hyper_grid %>% 
    dplyr::arrange(OOB_RMSE) %>% 
    head(10)

## The best model is mtry = 28, node_size = 3, sample = 0.8
## Confirm that this the model performance is stable with this setup
OOB_RMSE = vector(mode = "numeric", length = 100)
for (i in seq_along(OOB_RMSE)){
    optimal_ranger = ranger(formula         = Sale_Price ~.,
                            data            = ames_train,
                            num.trees       = 500,
                            mtry            = 28,
                            min.node.size   = 3,
                            sample.fraction = 0.8)
    OOB_RMSE[i] = sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20, col = "dodgerblue1")
