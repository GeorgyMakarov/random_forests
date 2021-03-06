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


# EDA ---------------------------------------------------------------------

# Split numerical and non-numerical variables
df_num = train[, c("Age", "SibSp", "Parch", "Fare")]
df_cat = train[, c("Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked")]


# Make summary of numerical variables
# We see that there are NA values in Age -- if this is true for the train set
# this might be also true for the test set. The number of NA is significant,
# so consider different options how to impute them.
summary(df_num)


# Make histograms of all num vars to check normality of distribution
# The histograms show that Age resembles normal distribution. Parch, SibSp is 
# a cat vars. They are just integers that represent family size. It is natural
# that smaller size family is more common than the larger one. Consider 
# making FamSize = SibSp + Parch + 1 cat variable. The +1 is required to count
# for the travelling person itself. Fare is heavily skewed to the left. It might
# be a good idea to normalize Fare.
for (i in 1:ncol(df_num)){
    hist(df_num[, i],
         main = colnames(df_num)[i],
         col  = "dodgerblue1")
}


# Find correlation between numeric variables
# This is a good point to reduce the number of variables for the random forest
# The plot shows that there is no significant correlation between numeric vars
apply(df_num, 2, FUN = function(x){sum(is.na(x))})
df_num = na.omit(df_num)
corrplot::corrplot(corr = cor(df_num), type = "upper", addCoef.col = "black")


# Compare survival rates across numeric variables
# Survival rates show that younger people had greater chance of survival. Rich
# also had more chances to survive compared to the poor. Larger families with
# both parents and children had greater chance of survival. At the same time,
# increase in the number of Siblings reduced the chance of survival.
train %>% 
    group_by(Survived) %>% 
    summarise(Age   = mean(Age, na.rm = T),
              Fare  = mean(Fare, na.rm = T),
              Parch = mean(Parch, na.rm = T),
              SibSp = mean(SibSp, na.rm = T))


# Make bar plots of categorical variables
# The plot shows that the dataset is imbalanced to those who did not survive.
# Majority of the passengers belonged to 3rd class cabin. There were more male
# passengers than there were females. There are too many different types of 
# tickets so we can't really make any suggestion on this one. The plot also 
# shows that people with no cabin prevale over other types. Majority of 
# passengers embarked in Southampton and there are some passenger wich we
# do not know where they'd embarked.
for (i in 1:ncol(df_cat)){
    tmp = 
        df_cat %>% 
        group_by(df_cat[, i]) %>% 
        summarise(Count = n())
    colnames(tmp)[1] = "x"
    barplot(height    = tmp$Count, 
            col       = "dodgerblue1",
            main      = paste("Barplot of", colnames(df_cat)[i]),
            names.arg = unique(tmp$x))
    rm(tmp, i)
}


# Make tables to compare categorical variables by survival rate
# Passengers from 1st class had almost 2.5 times more chances to survived
# compared to 3rd class and some 20% more chances to survive compared to 2nd
# class.
prop.table(table(df_cat$Pclass, df_cat$Survived), 1)


# Female passengers had much greater chance of survival
# It is possible to split those who survived based on Sex and Pclass
prop.table(table(df_cat$Sex, df_cat$Survived), 1)
aggregate(Survived ~ Pclass + Sex, 
          data = train, 
          FUN  = function(x) {sum(x) / length(x)})

# Passengers that embarked in Southampton had less chances to survive
# There are two persons that do not have embarkement place -- we need to think
# what we do about it. The aggregation table shows that it is possible to
# split the survived using Sex, Pclass and Embarked
prop.table(table(df_cat$Embarked, df_cat$Survived), 1)
aggregate(Survived ~ Pclass + Sex + Embarked,
          data = train,
          FUN  = function(x) {sum(x) / length(x)})
rm(df_cat, df_num)
rm(train, test)

# Feature engineering -----------------------------------------------------

# Key takeaways from EDA:
#  + convert Fare to groups
#  + predict Age NAs using model / median -- choose the best option
#  + create Age groups
#  + create FamSize = SibSp + Parch + 1 and make groups 'small', 'medium' etc.
#  + try Sex + Pclass + Embarked to predict survived


# Ideas from Cabin, Ticket and Title -- based on EDA
#  + create variable that reflects multiple cabins
#  + create TravelAlone variable based on FamSize
#  + extract Cabin letters
#  + create numeric ticket
#  + extract and transform titles


# Convert Fare to FareGroup categorical variable
# Histogram of Fare shows that Fare has great variation. We have to find such
# Fare groups so that they will be able to add split ability to our model.
apply(all_data, 2, FUN = function(x){sum(is.na(x))})
all_data$Fare[is.na(all_data$Fare) == T] = median(all_data$Fare, na.rm = T)
hist(all_data$Fare, col = "lightgreen")
fare_tmp = 
    all_data %>% filter(!is.na(Survived)) %>% select(Survived, Fare)
fare_tmp$FareGroup = round(fare_tmp$Fare, -1)
fare_tmp %>% 
    group_by(FareGroup) %>% 
    summarise(count = n(),
              surv  = sum(as.numeric(Survived), na.rm = T)) %>% 
    mutate(surv_rate = surv / count) %>% 
    ggplot2::ggplot() + geom_point(aes(x = FareGroup, y = surv_rate), 
                                   col = "darkgreen", alpha = 3/4)

# The plot shows that the optimal groups will be [0;10], (10;50], (50; 160],
# (160; 200], (200+)
all_data$FareGroup = "very cheap"
all_data$FareGroup[all_data$Fare > 10 & all_data$Fare <= 50] = "cheap"
all_data$FareGroup[all_data$Fare > 50 & all_data$Fare <= 160] = "medium"
all_data$FareGroup[all_data$Fare > 160 & all_data$Fare <= 200] = "expensive"
all_data$FareGroup[all_data$Fare > 200] = "very expensive"
all_data$FareGroup = as.factor(all_data$FareGroup)

prop.table(table(all_data$FareGroup[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)
rm(fare_tmp)


# Create a variable that extracts person's title
# Convert all military to Sir
# Convert all ladies to some kind of universal category
# Keep Kids separately -- might help with Age prediction
all_data$Title = sapply(all_data$Name, 
                        FUN = function(x){strsplit(x, split = '[,.]')[[1]][2]})
all_data$Title = sub(' ', '', all_data$Title)
table(all_data$Title[!is.na(all_data$Survived)])
prop.table(table(all_data$Title[!is.na(all_data$Survived)], 
                 all_data$Survived[!is.na(all_data$Survived)]), 1)


# Check how title and mean age correlate
# For the purpose of Age prediction we need the following groups by similar age:
# Master
# Miss = Miss + Mlle + Mme
# Ms
# Mr
# Mrs = Mrs + Jonkheer + Countess
# Dons = Dona + Don
# Lady
# Dr = Dr + Rev
# Militants = Sir + Major + Col + Capt
all_data %>% 
    group_by(Title) %>% 
    summarise(Age   = mean(Age, na.rm = T),
              Count = n()) %>% 
    ggplot2::ggplot() + geom_col(aes(x = Title, y = Age)) + coord_flip()


miss = c("Miss", "Mlle", "Mme")
mrs  = c("Mrs", "Jonkheer", "Countess")
dons = c("Dona", "Don")
dr   = c("Dr", "Rev")
mils = c("Sir", "Major", "Col", "Capt")

all_data$TtlAge = all_data$Title
all_data$TtlAge[all_data$Title %in% miss] = 'Miss'
all_data$TtlAge[all_data$Title %in% mrs]  = 'Mrs'
all_data$TtlAge[all_data$Title %in% dons] = 'Dons'
all_data$TtlAge[all_data$Title %in% dr]   = 'Dr'
all_data$TtlAge[all_data$Title %in% mils] = 'Mils'


# Check how to split Title by survival rate
# We can't keep the values that are present few times and we want to combine
# those with similar survival rate
all_data %>%
    filter(!is.na(Survived)) %>% 
    group_by(Title) %>% 
    summarise(count = n(),
              surv  = sum(as.numeric(Survived), na.rm = T)) %>% 
    mutate(surv_rate = surv / count)
    filter(count > 2) # add this to check fewer titles

mils = c("Capt", "Col", "Don", "Major", "Sir", "Rev")
lads = c("Jonkheer", "Lady", "Mlle", "Mme", "Ms", "the Countess")

all_data$Title[all_data$Title %in% mils] = "Mils"
all_data$Title[all_data$Title %in% lads] = "Lads"

all_data$Title  = as.factor(all_data$Title)
all_data$TtlAge = as.factor(all_data$TtlAge)

rm(dons, dr, lads, mils, miss, mrs)


# Convert cat to factor vars
all_data$Pclass   = as.factor(all_data$Pclass)
all_data$Sex      = as.factor(all_data$Sex)
all_data$Embarked = as.factor(all_data$Embarked)


# Check correlation of Age and categorical variables in order to be
# able to predict Age using a model. Use one way ANOVA tests. ANOVA test shows
# that all categorical variables are correlated to Age
age_temp = all_data %>% select(Age, Pclass, Sex, Embarked, FareGroup, TtlAge)
df_age   = na.omit(age_temp)
anova(lm(Age ~., data = df_age))

set.seed(123)
age_intrain = createDataPartition(df_age$Age, p = 0.8, list = F)
age_train   = df_age[age_intrain, ]
age_test    = df_age[-age_intrain,]

agesh_grid = expand.grid(mtry        = seq(1, 5, by = 1), 
                         node_size   = seq(2, 15, by = 1), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)

for (i in 1:nrow(agesh_grid)){
    
    model = ranger(formula         = Age ~.,
                   data            = age_train,
                   num.trees       = 500,
                   mtry            = agesh_grid$mtry[i],
                   min.node.size   = agesh_grid$node_size[i],
                   sample.fraction = agesh_grid$sample_size[i],
                   seed            = 123)
    
    agesh_grid$OOB_RMSE[i] = sqrt(model$prediction.error)
}

(age_choice = 
        agesh_grid %>% 
        dplyr::arrange(OOB_RMSE) %>% 
        head(10))

modela = ranger(formula         = Age ~.,
                data            = age_train,
                num.trees       = 500,
                mtry            = age_choice$mtry[1],
                min.node.size   = age_choice$node_size[1],
                sample.fraction = age_choice$sample_size[1],
                probability     = F, 
                seed            = 123)

modela

age_pred = predict(modela, data = age_test, type = "response")$predictions
error    = age_test$Age - round(age_pred, 0)
rmse_out = sqrt(mean(error^2))
rmse_out


# Check if a model is any better than using a median
# The model is better than the median
age_median = rep(median(age_test$Age), nrow(age_test))
err_median = age_test$Age - age_median
rmse_med   = sqrt(mean(err_median^2))
rmse_med


# Make prediction for NA values in Age
all_age_pred = predict(modela, data = age_temp, type = "response")$predictions
all_age_pred = round(all_age_pred, 0)

all_data$age_temp = all_age_pred
all_data$Age[is.na(all_data$Age)] = all_data$age_temp[is.na(all_data$Age)]
all_data = all_data %>% select(-age_temp)


# Split age into 3 groups
all_data$AgeGroup = "young"
all_data$AgeGroup[all_data$Age > 18 & all_data$Age < 60] = "adult"
all_data$AgeGroup[all_data$Age >= 60] = "elder"


# Check survival rates across the groups
prop.table(table(all_data$AgeGroup[!is.na(all_data$Survived)], 
                 all_data$Survived[!is.na(all_data$Survived)]), 1)
rm(age_choice, age_intrain, age_temp, age_test, age_train, agesh_grid,
   df_age, model, modela, age_median, age_pred, all_age_pred, err_median,
   error, i, rmse_med, rmse_out)


# Create FamSize variable
# Create TravAlone if FamSize == 1
all_data$FamSize = all_data$SibSp + all_data$Parch + 1
summary(all_data$FamSize)
all_data$TravAlone = "no"
all_data$TravAlone[all_data$FamSize == 1] = "yes"
all_data$TravAlone = as.factor(all_data$TravAlone)


# Check survival rates if a person travelled alone
# The table shows that those who travelled alone had less chance to survive
# compared to those who travelled with family
prop.table(table(all_data$TravAlone[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)


# Convert FamSize to FamGroup
# The table shows that alone travellers represent the most part of the
# passengers. We have to keep them separate. Families of size 2-3 had similar
# chances to survive -- we can call it `small` family. The family of 4 is 
# representative enough with 29 observations and has extremely high survival
# rate of 0.72 -- we want to keep them separately as 'medium'. Families with
# more than 5 persons had much lower chance of survival -- we can keep them
# in the 'large' group.
table(all_data$FamSize[!is.na(all_data$Survived)])
prop.table(table(all_data$FamSize[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)
all_data$FamGroup = "alone"
all_data$FamGroup[all_data$FamSize >= 2 & all_data$FamSize <= 3] = "small"
all_data$FamGroup[all_data$FamSize == 4] = "medium"
all_data$FamGroup[all_data$FamSize >= 5] = "large"
prop.table(table(all_data$FamGroup[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)


# Create variable standing for multiple cabins
# Count the number of cabins and return it as a variable. The table shows that
# people with higher number of cabins had greater chance of survival. However
# the count of people with more than 2 cabins is relatively small. Convert
# to CabinGroups.
tmp = strsplit(all_data$Cabin, " ")
tmp = lapply(tmp, function(x) length(unlist(x)))
all_data$CabinMult = unlist(tmp)
table(all_data$CabinMult)
prop.table(table(all_data$CabinMult[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)

all_data$cabin_temp = "nocabin"
all_data$cabin_temp[all_data$CabinMult == 1] = "single"
all_data$cabin_temp[all_data$CabinMult >= 2] = "multiple"
table(all_data$cabin_temp)
prop.table(table(all_data$cabin_temp[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)

all_data$CabinMult = all_data$cabin_temp
all_data = all_data %>% select(-cabin_temp)
all_data$CabinMult = as.factor(all_data$CabinMult)


# Create factor variable that stands for cabin letter. N stands for none.
# The table shows that passengers from cabins with letters B to F had greater
# chance of survival. This allows us to suggest that cabin letter is a good way
# of predicting survival.
tmp = all_data$Cabin
tmp = lapply(tmp, grep, pattern = "[[:alpha:]]")
tmp[lapply(tmp, length) == 0] = 0
tmp = unlist(tmp)
all_data$CabinLet = substr(x = all_data$Cabin, start = tmp, stop = tmp)
all_data$CabinLet[all_data$CabinLet == ""] = "N"


# The table shows that the count of some letters is fairly small. This will
# not provide us any helpful data. It might be better to convert to HasLetter
# variable for cabins with letters.
table(all_data$CabinLet)
prop.table(table(all_data$CabinLet[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)

all_data$cabin_letter = "no"
all_data$cabin_letter[!(all_data$CabinLet %in% "N")] = "yes"
table(all_data$cabin_letter)
prop.table(table(all_data$cabin_letter[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)
all_data$CabinLet = all_data$cabin_letter
all_data = all_data %>% select(-cabin_letter)
rm(tmp)


# Create factor variable if ticket contained numeric values: 0 - non-numeric,
# 1 - numeric ticket.
# The table shows that the proportion of survived for tickets that contain
# letters and numeric tickets is the same.
tmp = lapply(all_data$Ticket, grep, pattern = "[[:alpha:]]")
tmp[lapply(tmp, length) > 0]  = 0
tmp[lapply(tmp, length) == 0] = 1
all_data$TicketNum = unlist(tmp)
all_data$TicketNum = factor(ifelse(all_data$TicketNum == 1, "yes", "no"),
                            levels = c("yes", "no"))
prop.table(table(all_data$TicketNum[!is.na(all_data$Survived)],
                 all_data$Survived[!is.na(all_data$Survived)]), 1)
rm(tmp)


# Key takeaways from feature engineering
# Features to take into account:
# Sex + Pclass + Embarked + FareGroup + Title + AgeGroup + TravAlone +
# FamGroup + CabinMult + CabinLet
# Features we are not sure about:
# Age + SibSp + Parch + Fare + Cabin + FamSize


# Data preprocessing ------------------------------------------------------

# Action plan:
#  + select relevant features
#  + convert all features to factors
#  - check for correlated features
#  - check for outliers
#  - make balanced dataset

sel_data = all_data %>% select(Survived, Sex, Pclass, Embarked, FareGroup,
                               Title, AgeGroup, TravAlone, FamGroup, CabinMult,
                               CabinLet, Segm)

sel_data$AgeGroup = as.factor(sel_data$AgeGroup)
sel_data$FamGroup = as.factor(sel_data$FamGroup)
sel_data$CabinLet = as.factor(sel_data$CabinLet)






# TODO: check for correlated variables
# TODO: check for outliers
# TODO: make balanced dataset for training


training = sel_data %>% filter(Segm == "training") %>% select(-Segm)
testing  = sel_data %>% filter(Segm == "testing") %>% select(-Segm)
training$Survived = 
    factor(ifelse(training$Survived == 0, "no", "yes"),
           levels = c("yes", "no"))


# Model training ----------------------------------------------------------

# Action plan:
#  + no model -- all died
#  + no model -- all women survived
#  + random forest1: Sex + Pclass + Embarked


# No model predict that all died
# The accuracy is 0.622
pred_all_died = data.frame(pred = rep(0, 418))
pred_all_died = factor(ifelse(pred_all_died$pred == 1, "yes", "no"), 
                       levels = c("yes", "no"))
output_all_d  = data.frame(obs = ans$Survived, pred = pred_all_died)
confusionMatrix(data      = output_all_d$pred, 
                reference = output_all_d$obs)
rm(pred_all_died, output_all_d)


# No model predict that all female survived
# The accuracy is 0.766
test_fems = testing
test_fems$Survived = "no"
test_fems$Survived[test_fems$Sex == "female"] = "yes"
test_fems$Survived = factor(test_fems$Survived, levels = c("yes", "no"))

output_fems = data.frame(obs = ans$Survived, pred = test_fems$Survived)
confusionMatrix(data      = output_fems$pred, 
                reference = output_fems$obs)
rm(test_fems, output_fems)


# Random forests 
# Sex + Pclass + Embarked             = 0.7775
hyper_grid = expand.grid(mtry        = seq(1, 4, by = 1), 
                         node_size   = seq(2, 15, by = 2), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)

for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = Survived ~ Sex + Pclass + Embarked + AgeGroup,
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


# Train tuned model
modelb = ranger(formula         = Survived ~ Sex + Pclass + Embarked + AgeGroup,
                data            = training,
                num.trees       = 500,
                mtry            = choice$mtry[1],
                min.node.size   = choice$node_size[1],
                sample.fraction = choice$sample_size[1],
                probability     = T, 
                seed            = 123)

pred = predict(modelb, data = testing, type = "response")$predictions
rm(model, hyper_grid, choice, modelb, i)

output      = data.frame(obs  = ans$Survived,
                         pred = pred)
output$pred = factor(ifelse(output$pred.yes >= output$pred.no, "yes", "no"),
                     levels = c("yes", "no"))
output      = output %>% select(obs, pred, yes = pred.yes, no = pred.no)

confusionMatrix(data = output$pred, reference = output$obs)
twoClassSummary(data = output, lev  = levels(output$obs))
rm(output, pred)

