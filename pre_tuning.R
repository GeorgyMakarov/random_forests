suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ROSE))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(ranger))
suppressPackageStartupMessages(library(caret))


# Prepare raw data
data("infert")
infert$case[infert$case == 0] = 2
infert$case[infert$case == 1] = "yes"
infert$case[infert$case == 2] = "no"
infert$case = factor(x      = infert$case,
                     levels = c("yes", "no"))

set.seed(123)
in_train = createDataPartition(y     = infert$case,
                               times = 1,
                               p     = 0.8,
                               list  = F)
infert_train = infert[in_train,]
infert_test  = infert[-in_train,]
rm(in_train)


# Exploratory data analysis
summary(infert_train)
sum(complete.cases(infert_train))


# Scatter plots show that there is a correlation between stratum and
# pooled.stratum variables. Other variables do not correlate.
temp1 = infert_train[, c(2:5)]
temp2 = infert_train[, c(5:8)]

featurePlot(x    = temp1[, -4],
            y    = temp1$case,
            plot = 'ellipse',
            auto.key = list(columns = 2))

featurePlot(x    = temp2[, -1],
            y    = temp2$case,
            plot = 'ellipse',
            auto.key = list(columns = 2))

rm(temp1, temp2)


# Correlation matrix confirms that there is significant correlation between
# stratum and pooled.stratum -- we might not need both variables to run the
# the model in the future
corrplot::corrplot(corr = cor(infert_train[, -c(1, 5)]),
                   type = "upper",
                   addCoef.col = "black")

infert_train = infert_train[, -8]


# Boxplot comparison shows that the main variable that differ yes/no is
# spontaneous. Other variables are almost the same. There also outliers in
# the parity variable
featurePlot(x    = infert_train[, -c(1, 5)],
            y    = infert_train$case,
            plot = "box",
            scales = list(y = list(relation = "free"),
                          x = list(rot      = 90)),
            auto.key = list(columns = 2))


# Check outliers using Mahalanobis Distance
# Substitute outliers with median values
psych::pairs.panels(infert_train[, -c(1, 5)], stars = T)
temp_train = infert_train[, -c(1, 5)]
mahal_dist = mahalanobis(temp_train, colMeans(temp_train), cov(temp_train))
plot(x     = mahal_dist,
     col   = "dodgerblue",
     pch   = 19,
     frame = F,
     main  = "Mahalanobis distances")
abline(h   = 10.5,
       col = "red",
       lty = 2)

outs = which(mahal_dist > 10.5)
rm(mahal_dist, temp_train)

infert_train$parity[outs] = median(infert_train$parity)

# Check how deleting of correlated pred impacts the performance of the model
# The test shows that deleting correlated variables increased ROC from 0.808
# to 0.822
## mMake balanced dataset
temp_balance = ROSE(case ~., data = infert_train, seed = 1)$data
temp_balance$case = relevel(temp_balance$case, ref = "yes")
prop.table(table(temp_balance$case))

## search grid: mtry = 5, ns = 6, sample = 0.7
hyper_grid_bs = expand.grid(mtry        = seq(2, 6, by = 1),
                            node_size   = seq(2, 9, by = 2),
                            sample_size = c(0.55, 0.632, 0.70, 0.80),
                            OOB_RMSE    = 0)

for (i in 1:nrow(hyper_grid_bs)){
    
    model = ranger(formula         = case ~.,
                   data            = temp_balance,
                   num.trees       = 500,
                   mtry            = hyper_grid_bs$mtry[i],
                   min.node.size   = hyper_grid_bs$node_size[i],
                   sample.fraction = hyper_grid_bs$sample_size[i],
                   seed            = 123)
    
    hyper_grid_bs$OOB_RMSE[i] = sqrt(model$prediction.error)
}

hyper_grid_bs %>% 
    dplyr::arrange(OOB_RMSE) %>% 
    head(10)

## make model with best params
model6 = ranger(formula         = case ~.,
                data            = temp_balance,
                num.trees       = 500,
                mtry            = 3,
                min.node.size   = 6,
                sample.fraction = 0.550,
                probability     = T, 
                seed            = 123)

## make prediction using the model with reduced variables
output6 = data.frame(obs  = infert_test$case,
                     pred = predict(model6,
                                    data = infert_test,
                                    type = "response")$predictions)



output6$pred = "yes"
output6 = output6 %>% select(obs, pred, pred.yes, pred.no)
output6$pred[output6$pred.yes < output6$pred.no] = "no"
colnames(output6) = c("obs", "pred", "yes", "no")
output6$pred = factor(x = output6$pred,
                      levels = c("yes", "no"))


confusionMatrix(data      = output6$pred,
                reference = output6$obs)
twoClassSummary(data = output6,
                lev  = levels(output6$obs))

# Plot the ROC curve
roc_curve = pROC::roc(response  = output6$obs,
                      predictor = output6$pred)


# You do not need to take care of near zero variance because tree-based models
# do not depend on NZV. However it might be good to scale and center the data
# -- lets test if it is any good to the model. The test shows that scaling and
# centering does not help to improve the model performance.
transformed = preProcess(x      = infert_train[, -5],
                         method = c("center", "scale", "YeoJohnson", "nzv"))
transformed

trans_train = predict(transformed, infert_train)
trans_test  = predict(transformed, infert_test)

trans_bs = ROSE(case ~., data = trans_train, seed = 1)$data
trans_bs$case = relevel(trans_bs$case, ref = "yes")
prop.table(table(trans_bs$case))

for (i in 1:nrow(hyper_grid_bs)){
    
    model = ranger(formula         = case ~.,
                   data            = trans_bs,
                   num.trees       = 500,
                   mtry            = hyper_grid_bs$mtry[i],
                   min.node.size   = hyper_grid_bs$node_size[i],
                   sample.fraction = hyper_grid_bs$sample_size[i],
                   seed            = 123)
    
    hyper_grid_bs$OOB_RMSE[i] = sqrt(model$prediction.error)
}

hyper_grid_bs %>% 
    dplyr::arrange(OOB_RMSE) %>% 
    head(10)

model8 = ranger(formula         = case ~.,
                data            = trans_bs,
                num.trees       = 500,
                mtry            = 5,
                min.node.size   = 6,
                sample.fraction = 0.632,
                probability     = T, 
                seed            = 123)
model8

output8 = data.frame(obs  = trans_test$case,
                     pred = predict(model8,
                                    data = trans_test,
                                    type = "response")$predictions)
output8$pred = "yes"
output8 = output8 %>% select(obs, pred, pred.yes, pred.no)
output8$pred[output8$pred.yes < output8$pred.no] = "no"
colnames(output8) = c("obs", "pred", "yes", "no")
output8$pred = factor(x = output8$pred,
                      levels = c("yes", "no"))


confusionMatrix(data      = output8$pred,
                reference = output8$obs)
twoClassSummary(data = output8,
                lev  = levels(output8$obs))






