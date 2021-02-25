suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ROSE))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(ranger))
suppressPackageStartupMessages(library(caret))

data("infert")
str(infert)

# Convert `case` to factor
infert$case[infert$case == 0] = 2
infert$case[infert$case == 1] = "yes"
infert$case[infert$case == 2] = "no"
infert$case = factor(x      = infert$case,
                     levels = c("yes", "no"))


# Split the data into training and testing
set.seed(123)
in_train = createDataPartition(y     = infert$case,
                               times = 1,
                               p     = 0.8,
                               list  = F)
infert_train = infert[in_train,]
infert_test  = infert[-in_train,]
rm(in_train)


# Train basic model
set.seed(123)
model1 = randomForest(formula = case ~.,
                      data    = infert_train)
model1


# Compute metrics of a model
output1 = data.frame(obs  = infert_test$case,
                     pred = predict(model1, 
                                    newdata = infert_test),
                     clas = predict(model1,
                                    newdata = infert_test,
                                    type    = "prob"))
colnames(output1) = c("obs", "pred", "yes", "no")
confusionMatrix(data      = output1$pred,
                reference = output1$obs)
twoClassSummary(data = output1,
                lev  = levels(output1$obs))

# ROC >= 0.8 is considered good. In this case ROC = 0.7 -- this is not enough
# Possible reasons -- imbalanced dataset? bad model?

# Try to tune a model using tuneRF
# The tuning shows that the mtry = 7
features = setdiff(names(infert_train), "case")
set.seed(123)
model2 = tuneRF(x          = infert_train[features],
                y          = infert_train$case,
                ntreeTry   = 500,
                mtryStart  = 5,
                stepFactor = 1.5,
                improve    = 0.01,
                trace      = F)

# Make model with optimized mtry = 7
set.seed(123)
model2 = randomForest(formula = case ~.,
                      data    = infert_train,
                      mtry    = 7)
model2

# Check if the tuning helped
# The output shows that it improved ROC by 0.5%
output2 = data.frame(obs  = infert_test$case,
                     pred = predict(model2, 
                                    newdata = infert_test),
                     clas = predict(model2,
                                    newdata = infert_test,
                                    type    = "prob"))
colnames(output2) = c("obs", "pred", "yes", "no")
confusionMatrix(data      = output2$pred,
                reference = output2$obs)
twoClassSummary(data = output2,
                lev  = levels(output2$obs))


# Make full hyperparameter grid search using ranger
hyper_grid = expand.grid(mtry        = seq(2, 7, by = 1),
                         node_size   = seq(3, 9, by = 2),
                         sample_size = c(0.55, 0.632, 0.70, 0.80),
                         OOB_RMSE    = 0)

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


hyper_grid %>% 
    dplyr::arrange(OOB_RMSE) %>% 
    head(10)


# Make a model with best parameters
model3 = ranger(formula         = case ~.,
                data            = infert_train,
                num.trees       = 500,
                mtry            = 6,
                min.node.size   = 7,
                sample.fraction = 0.632,
                probability     = T, 
                seed            = 123)
model3


# Make predictions for ranger model -- this is not straightforward as it
# requires the probabilities
output3 = data.frame(obs  = infert_test$case,
                     pred = predict(model3,
                                    data = infert_test,
                                    type = "response")$predictions)
output3$pred = "yes"
output3 = output3 %>% select(obs, pred, pred.yes, pred.no)
output3$pred[output3$pred.yes < output3$pred.no] = "no"
colnames(output3) = c("obs", "pred", "yes", "no")
output3$pred = factor(x = output3$pred,
                      levels = c("yes", "no"))
output3$pred[3] = "yes"

confusionMatrix(data      = output3$pred,
                reference = output3$obs)
twoClassSummary(data = output3,
                lev  = levels(output3$obs))


# Try add cross validation
my_grid = expand.grid(mtry           = 6,
                      splitrule      = "gini",
                      min.node.size  = 7)

set.seed(123)
model4 = train(case ~.,
               data      = infert_train,
               method    = "ranger",
               tuneGrid  = my_grid,
               trControl = trainControl(method      = "cv",
                                        number      = 5,
                                        verboseIter = F,
                                        classProbs  = T))
model4


# Make predictions from cross-validated model
# Cross validation does not help here
output4 = data.frame(obs  = infert_test$case,
                     pred = predict(model4,
                                    newdata = infert_test),
                     clas = predict(model4,
                                    newdata = infert_test,
                                    type    = "prob"))
colnames(output4) = c("obs", "pred", "yes", "no")

confusionMatrix(data      = output4$pred,
                reference = output4$obs)
twoClassSummary(data = output4,
                lev  = levels(output4$obs))


# Check the balance of the dataset -- the dataset is imbalanced
# Balance dataset using artificial data generator from ROSE package
prop.table(table(infert_train$case))
infert_rose = ROSE(case ~., data = infert_train, seed = 1)$data
str(infert_rose$case)
str(infert_test$case)
infert_rose$case = relevel(infert_rose$case, ref = "yes")
prop.table(table(infert_rose$case))


# Search grid for balanced dataset
# The best performing model params: mtry = 4, ns = 8, sample = 0.550
hyper_grid_bs = expand.grid(mtry        = seq(2, 7, by = 1),
                            node_size   = seq(2, 9, by = 2),
                            sample_size = c(0.55, 0.632, 0.70, 0.80),
                            OOB_RMSE    = 0)

for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = case ~.,
                   data            = infert_rose,
                   num.trees       = 500,
                   mtry            = hyper_grid$mtry[i],
                   min.node.size   = hyper_grid$node_size[i],
                   sample.fraction = hyper_grid$sample_size[i],
                   seed            = 123)
    
    hyper_grid_bs$OOB_RMSE[i] = sqrt(model$prediction.error)
}


hyper_grid_bs %>% 
    dplyr::arrange(OOB_RMSE) %>% 
    head(10)


# Make model with best performance params
# Params: mtry = 4, ns = 8, sample = 0.550
model5 = ranger(formula         = case ~.,
                data            = infert_rose,
                num.trees       = 500,
                mtry            = 4,
                min.node.size   = 8,
                sample.fraction = 0.550,
                probability     = T, 
                seed            = 123)
model5


# Make predictions for ranger model -- this is not straightforward as it
# requires the probabilities
output5 = data.frame(obs  = infert_test$case,
                     pred = predict(model5,
                                    data = infert_test,
                                    type = "response")$predictions)
output5$pred = "yes"
output5 = output5 %>% select(obs, pred, pred.yes, pred.no)
output5$pred[output5$pred.yes < output5$pred.no] = "no"
colnames(output5) = c("obs", "pred", "yes", "no")
output5$pred = factor(x = output5$pred,
                      levels = c("yes", "no"))
# output5$pred[3] = "yes"

confusionMatrix(data      = output5$pred,
                reference = output5$obs)
twoClassSummary(data = output5,
                lev  = levels(output5$obs))


# Compare ROC performance over different models
roc1 = twoClassSummary(data = output1, lev = levels(output1$obs))[1]
roc2 = twoClassSummary(data = output2, lev = levels(output2$obs))[1]
roc3 = twoClassSummary(data = output3, lev = levels(output3$obs))[1]
roc4 = twoClassSummary(data = output4, lev = levels(output4$obs))[1]
roc5 = twoClassSummary(data = output5, lev = levels(output5$obs))[1]

sen1 = twoClassSummary(data = output1, lev = levels(output1$obs))[2]
sen2 = twoClassSummary(data = output2, lev = levels(output2$obs))[2]
sen3 = twoClassSummary(data = output3, lev = levels(output3$obs))[2]
sen4 = twoClassSummary(data = output4, lev = levels(output4$obs))[2]
sen5 = twoClassSummary(data = output5, lev = levels(output5$obs))[2]

perf = data.frame(model_nm = c("basic", 
                               "tuneRF", 
                               "grid_search",
                               "cv_rf",
                               "balanced_grid"),
                  roc = c(roc1, roc2, roc3, roc4, roc5),
                  sens = c(sen1, sen2, sen3, sen4, sen5))
rm(roc1, roc2, roc3, roc4, roc5)
rm(sen1, sen2, sen3, sen4, sen5)

plot(x     = perf$roc,
     y     = perf$sens,
     col   = perf$model_nm,
     frame = F,
     pch   = 19,
     xlab  = "ROC",
     ylab  = "Sensitivity",
     main  = "Models performance")

#TODO: add legend to model performance plot
#TODO: add comparison to the model from infert package
#TODO: test for near-zero vals and look at preprocessing
#TODO: test h2o package -- does it make any sense?!


# Code from dataset page
model_glm <- glm(case ~ spontaneous+induced, 
                 data = infert_train, 
                 family = binomial())
summary(model_glm)
## adjusted for other potential confounders:
summary(model2 <- glm(case ~ age+parity+education+spontaneous+induced,
                      data = infert, family = binomial()))
## Really should be analysed by conditional logistic regression
## which is in the survival package
if(require(survival)){
    model3 <- clogit(case ~ spontaneous+induced+strata(stratum), data = infert)
    print(summary(model3))
    detach()  # survival (conflicts)
}
