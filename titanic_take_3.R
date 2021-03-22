# This script predicts Titanic's survival using different models from
# simple models to more complicated models. The purpose of this script is to
# learn random forest tuning and acheive maximum possible score.
setwd("/home/georgy/Документы/Rstudio/sreda_solutions/interview_tasks/task_3")
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(ROSE))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(ranger))
suppressPackageStartupMessages(library(caret))


# Read and prepare raw data -----------------------------------------------

train = read.csv("train.csv", stringsAsFactors = FALSE)
test  = read.csv("test.csv", stringsAsFactors = FALSE)
ans   = read.csv("correct_answers.csv", stringsAsFactors = FALSE)


# Combine train and test data -- this is helpful for feature engineering
# Add Survived to test as NA, change the order of columns
# Check that column names match before merging
train$Segm    = "training"
test$Segm     = "testing"
test$Survived = NA

test = 
    test %>% 
    select(PassengerId, Survived, everything())

colnames(test) == colnames(train)
all_data = rbind(train, test)


# Make Survived in answers a cat var
ans$Survived = factor(ifelse(ans$Survived == "yes", "yes", "no"),
                      levels = c("yes", "no"))
rm(train, test)


# Impute missing values
all_data$Fare[is.na(all_data$Fare) == T]   = median(all_data$Fare, na.rm = T)
all_data$Age[is.na(all_data$Age) == T]     = median(all_data$Age, na.rm = T)
all_data$Embarked[all_data$Embarked == ""] = "S"


# Check correlation between variables
# There is strong correlation between Pclass and Fare
# There is strong correlation between SibSp and Parch
temp_cor =
    all_data %>% 
    filter(Segm == "training") %>% 
    select(c(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)) %>%  
    mutate(Embarked = as.numeric(as.factor(Embarked)), 
           Sex      = as.numeric(as.factor(Sex)))
corrplot::corrplot(corr = cor(temp_cor), type = "upper", addCoef.col = "black")
rm(temp_cor)


# Round Age to full years
all_data$Age = round(all_data$Age, 0)


# The data shows that 52% of the boys under 16 survived
# Make a feature that separates boys from other men
all_data$IsBoy = "no"
all_data$IsBoy[all_data$Sex == "male" & all_data$Age < 16] = "yes"


# Add title as a factor
all_data$Title <- sapply(all_data$Name, 
                         FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
all_data$Title = sub(' ', '', all_data$Title)
all_data$Title[all_data$Title %in% c('Mme', 'Mlle')]  = 'Mlle'
all_data$Title[all_data$Title %in% c('Capt', 'Don', 
                                     'Major', 'Sir')] = 'Sir'
all_data$Title[all_data$Title %in% c('Dona', 'Lady', 
                                     'the Countess', 'Jonkheer')] = 'Lady'




# Make model --------------------------------------------------------------

# select features
sel_data = 
    all_data %>% 
    select(Survived, Sex, Pclass, Embarked, IsBoy, Title, Segm)

sel_data$Sex       = as.factor(sel_data$Sex)
sel_data$Pclass    = as.factor(sel_data$Pclass)
sel_data$Embarked  = as.factor(sel_data$Embarked)
sel_data$IsBoy     = as.factor(sel_data$IsBoy)
sel_data$Title     = as.factor(sel_data$Title)

training = sel_data %>% filter(Segm == "training") %>% select(-Segm)
testing  = sel_data %>% filter(Segm == "testing") %>% select(-Segm)

training$Survived = factor(ifelse(training$Survived == 0, "no", "yes"), 
                           levels = c("yes", "no"))
rm(sel_data)


hyper_grid = expand.grid(mtry        = seq(1, ncol(training) - 1, by = 1), 
                         node_size   = seq(2, 15, by = 2), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)

for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = Survived ~ .,
                   data            = training,
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


modelb = ranger(formula         = Survived ~ .,
                data            = training,
                num.trees       = 500,
                mtry            = choice$mtry[1],
                min.node.size   = choice$node_size[1],
                sample.fraction = choice$sample_size[1],
                probability     = T, 
                seed            = 123)

pred = predict(modelb, data = testing, type = "response")$predictions
rm(model, hyper_grid, choice, i)

output      = data.frame(obs  = ans$Survived,
                         pred = pred)
output$pred = factor(ifelse(output$pred.yes >= output$pred.no, "yes", "no"),
                     levels = c("yes", "no"))
output      = output %>% select(obs, pred, yes = pred.yes, no = pred.no)

confusionMatrix(data = output$pred, reference = output$obs)
twoClassSummary(data = output, lev  = levels(output$obs))
rm(output, pred)