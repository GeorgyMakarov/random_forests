# This script predicts Titanic's survival using different models from
# simple models to more complicated models. The purpose of this script is to
# learn random forest tuning and acheive maximum possible score.
setwd("/home/georgy/Документы/Rstudio/sreda_solutions/interview_tasks/task_3")
suppressPackageStartupMessages(library(dplyr))
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


# Add title as a factor
# Convert titles to men, women, kids
all_data$Title <- sapply(all_data$Name, 
                         FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
all_data$Title = sub(' ', '', all_data$Title)
men   = c("Capt", "Don", "Major", "Col", "Rev", "Dr", "Sir", "Mr", "Jonkheer")
women = c("Dona", "the Countess", "Mme", "Mlle", "Ms", "Miss", "Lady", "Mrs")
all_data$Title[all_data$Title %in% men]      = "man"
all_data$Title[all_data$Title %in% women]    = "woman"
all_data$Title[all_data$Title %in% "Master"] = "boy"
rm(men, women)


# Make GroupId as concatenation of Surname + Pclass + Ticket + Fare + Embarked
# Use this to identify family groups or mothers travelling with children
all_data$Surname = substring(all_data$Name, 0, regexpr(",",all_data$Name)-1)
all_data$GroupId = paste(all_data$Surname,
                         all_data$Pclass,
                         sub('.$', 'X', all_data$Ticket),
                         all_data$Fare,
                         all_data$Embarked,
                         sep = "-")


# Identify relatives that travelled together but had different Surnames
# This is helpful as they most likely all lived or all died together
all_data$GroupId[all_data$Title == "man"] = 'noGroup'
all_data$GroupFreq = ave(1:nrow(all_data), all_data$GroupId, FUN = length)
all_data$GroupId[all_data$GroupFreq <= 1] = 'noGroup'
all_data$TicketId = paste(all_data$Pclass,
                          sub('.$', 'X', all_data$Ticket),
                          all_data$Fare,
                          all_data$Embarked,
                          sep = "-")
for (i in which(all_data$Title != "man" & all_data$GroupId == "noGroup")){
    all_data$GroupId[i] = 
        all_data$GroupId[all_data$TicketId == all_data$TicketId[i]][1]
}
rm(i)


# Add survival rates by GroupIds
all_data$GroupSurvival        = NA
all_data$GroupSurvival[1:891] = ave(all_data$Survived[1:891],
                                    all_data$GroupId[1:891])
for (i in 892:1309){
    all_data$GroupSurvival[i] =
        all_data$GroupSurvival[
            which(all_data$GroupId == all_data$GroupId[i])[1]]
}


# Identify ids of the groups that where not identified properly
# Substitute non-identified groups Survival with 0 or 1
tmp = 
    all_data %>% 
    filter(is.na(GroupSurvival))


# Gibsons: Pclass = 1, Embarked = C, no Cabin -- predict 1
all_data$GroupSurvival[1260] = 1
all_data$GroupSurvival[1294] = 1


# Peacocks, Klasen, von Billard: Pclass =3, Embarked = S, no Cabin -- predict 0
perished = setdiff(tmp$PassengerId, c(1260, 1294))
all_data$GroupSurvival[all_data$PassengerId %in% perished] = 0
rm(perished)


# The data shows that 52% of the boys under 16 survived
# Make a feature that separates boys from other men
all_data$IsBoy = "no"
all_data$IsBoy[all_data$Sex == "male" & all_data$Age < 16] = "yes"


# Make model --------------------------------------------------------------

# Select features
sel_data = 
    all_data %>% 
    select(Survived, Sex, Pclass, Embarked,
           GroupId, GroupFreq, GroupSurvival, IsBoy, Segm)

sel_data$Sex       = as.factor(sel_data$Sex)
sel_data$Pclass    = as.factor(sel_data$Pclass)
sel_data$Embarked  = as.factor(sel_data$Embarked)
sel_data$IsBoy     = as.factor(sel_data$IsBoy)


training = sel_data %>% filter(Segm == "training") %>% select(-Segm)
testing  = sel_data %>% filter(Segm == "testing") %>% select(-Segm)

training$Survived = factor(ifelse(training$Survived == 0, "no", "yes"), 
                           levels = c("yes", "no"))
rm(sel_data)


hyper_grid = expand.grid(mtry        = seq(1, ncol(training) - 1, by = 1), 
                         node_size   = seq(2, 15, by = 1), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)

for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = Survived ~ .,
                   data            = training,
                   num.trees       = 200,
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
                num.trees       = 200,
                mtry            = choice$mtry[1],
                min.node.size   = choice$node_size[1],
                sample.fraction = choice$sample_size[1],
                probability     = T, 
                seed            = 123)

pred = predict(modelb, data = testing, type = "response")$predictions
rm(model, hyper_grid, choice, i)

output      = data.frame(obs  = ans$Survived,
                         pred = pred)
output$pred = factor(ifelse(output$pred.yes >= 0.55, "yes", "no"),
                     levels = c("yes", "no"))
output      = output %>% select(obs, pred, yes = pred.yes, no = pred.no)

confusionMatrix(data = output$pred, reference = output$obs)
twoClassSummary(data = output, lev  = levels(output$obs))
rm(output, pred)