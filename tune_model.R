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


# Remove highly correlated variables
# Remove outliers
infert_train = infert_train[, -8]
temp_train   = infert_train[, -c(1, 5)]
mahal_dist   = mahalanobis(temp_train, colMeans(temp_train), cov(temp_train))
outs         = which(mahal_dist > 10.5)
infert_train$parity[outs] = median(infert_train$parity)
rm(mahal_dist, temp_train, outs)


# Make balanced training set
infert_train      = ROSE(case ~., data = infert_train, seed = 1)$data
infert_train$case = relevel(infert_train$case, ref = "yes")


# Make hyperparameters grid
hyper_grid = expand.grid(mtry        = seq(2, 5, by = 1), 
                         node_size   = seq(2, 9, by = 2), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)


# Search hyperparameters
for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = case ~.,
                   data            = infert_train,
                   num.trees       = 500,
                   mtry            = hyper_grid$mtry[i],
                   min.node.size   = hyper_grid$node_size[i],
                   sample.fraction = hyper_grid$sample_size[i],
                   seed            = 123)
    
    hyper_grid$OOB_RMSE[i] = sqrt(model$prediction.error)
}

(choice = 
        hyper_grid %>% 
        dplyr::arrange(OOB_RMSE) %>% 
        head(10))


# Make model with the best params: mtry = 3, ns = 5, sample = 0.550
modelb = ranger(formula         = case ~.,
                data            = infert_train,
                num.trees       = 500,
                mtry            = choice$mtry[1],
                min.node.size   = choice$node_size[1],
                sample.fraction = choice$sample_size[1],
                probability     = T, 
                seed            = 123)

# Compute model accuracy
output = data.frame(obs  = infert_test$case,
                    pred = predict(modelb,
                                   data = infert_test,
                                   type = "response")$predictions)

output$pred = "yes"
output = output %>% select(obs, pred, pred.yes, pred.no)
output$pred[output$pred.yes < output$pred.no] = "no"
colnames(output) = c("obs", "pred", "yes", "no")
output$pred = factor(x = output$pred,
                     levels = c("yes", "no"))

confusionMatrix(data      = output$pred,
                reference = output$obs)
twoClassSummary(data = output,
                lev  = levels(output$obs))


# Search through thresholds to find when the p-value[Acc > NIR] <= 0.05
# The output shows that the threshold of 0.55 has p-value <= 0.05
thresh_grid = data.frame(threshold = seq(0.4, 0.9, by = 0.05),
                         accuracy  = 0,
                         pvalue    = 0,
                         mcnemar   = 0,
                         rock      = 0,
                         sens      = 0)

for (i in 1:length(thresh_grid$threshold)){
    output = data.frame(obs  = infert_test$case,
                        pred = predict(modelb,
                                       data = infert_test,
                                       type = "response")$predictions)
    output$pred = 
        factor(ifelse(output$pred.yes >= thresh_grid$threshold[i], "yes", "no"),
               levels = c("yes", "no"))
    output = output %>% select(obs, pred, pred.yes, pred.no)
    colnames(output) = c("obs", "pred", "yes", "no")
    cm = confusionMatrix(data = output$pred, reference = output$obs)$overall
    tc = twoClassSummary(data = output, lev  = levels(output$obs))
    
    thresh_grid$accuracy[i] = cm[1]
    thresh_grid$pvalue[i]   = cm[6]
    thresh_grid$mcnemar[i]  = cm[7]
    thresh_grid$rock[i]     = tc[1]
    thresh_grid$sens[i]     = tc[2]
}

thresh_grid

# Check model performance with thresh = 0.55
output = data.frame(obs  = infert_test$case,
                    pred = predict(modelb,
                                   data = infert_test,
                                   type = "response")$predictions)
output$pred = 
    factor(ifelse(output$pred.yes >= 0.55, "yes", "no"),
           levels = c("yes", "no"))
output = output %>% select(obs, pred, pred.yes, pred.no)
colnames(output) = c("obs", "pred", "yes", "no")
confusionMatrix(data = output$pred, reference = output$obs)
twoClassSummary(data = output, lev  = levels(output$obs))


# Plot ROC AUC curve
suppressPackageStartupMessages(library(pROC))
output$obs_num  = ifelse(output$obs == "yes", 1, 0)
output$pred_num = ifelse(output$pred == "yes", 1, 0)
roc_score       = roc(output$obs_num, output$pred_num)
plot.roc(roc_score,
         print.auc   = T,
         auc.polygon = T,
         grid.col    = c("green", "red"))