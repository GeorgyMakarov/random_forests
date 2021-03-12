# This script follows Ken Jee's video tutorial of titanic project. The purpose
# of this script is to learn random forest and data transformation required to
# improve random forest performance. I plan to use it as a basis for my car
# accident prediction project.
setwd("/home/georgy/Документы/Rstudio/sreda_solutions/interview_tasks/task_3")
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ROSE))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(ranger))
suppressPackageStartupMessages(library(caret))
seed = 123

train = read.csv("train.csv", stringsAsFactors = FALSE)
test  = read.csv("test.csv", stringsAsFactors = FALSE)

# Combine train and test data -- this is helpful for feature engineering
train$Segm    = "training"
test$Segm     = "testing"
test$Survived = NA

test = 
    test %>% 
    select(PassengerId, Survived, everything())

colnames(test) == colnames(train)
all_data = rbind(train, test)


# EDA ---------------------------------------------------------------------

# Understand the nature of the data. Make histograms and plots, count values and
# deal with missing data. Find correlated metrics. Test for outliers and
# make balanced dataset for training. Interesting hypothesis:
#   wealthy survived?
#   any impact from location?
#   old people buying more expensive tickets?
#   combine young and wealthy variables?
#   total spent?
# Feature engineering.

# General data overview
str(train)
summary(train)


# Split numerical and non-numerical variables
df_num = train[, c("Age", "SibSp", "Parch", "Fare")]
df_cat = train[, c("Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked")]


# Plot histograms of all numerical variables
# The plot shows that SibSp and Fare are not normal distribution. We might want
# to log transform them to normal distributions.
for (i in 1:ncol(df_num)){
    hist(df_num[, i],
         main = colnames(df_num)[i],
         col  = "dodgerblue1")
}


# Make correlation plot
# This does not show any significant correlation between the variables
corrplot::corrplot(corr = cor(df_num), type = "upper", addCoef.col = "black")


# Compare survival rates across numeric variables
# Survival rates show that younger people had greater chance of survival. Rich
# also had more chances to survived compared to the poor. Larger families with
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


# Make tables to compare categorical variables by survival
# Passengers of the 1st class had greater chance of survival. Passengers of 2nd
# class chance of survival decreased. Passengers from 3rd class had much less
# chance to survive compared to other classes. This might be relevant to our
# suggestion about rich survivals.
table(df_cat$Survived, df_cat$Pclass)

# Female passengers had much greater chance of survival.
table(df_cat$Survived, df_cat$Sex)

# Passengers that embarked in Southampton had less chances to survive
# There are two persons that do not have embarkement place -- we need to think
# what we do about it.
table(df_cat$Survived, df_cat$Embarked)
rm(df_cat, df_num)

# Feature engineering -----------------------------------------------------

# Create variable standing for multiple cabins
# Count the number of cabins and return it as a variable. The table shows that
# people with higher number of cabins had greater chance of survival.
tmp = strsplit(all_data$Cabin, " ")
tmp = lapply(tmp, function(x) length(unlist(x)))
all_data$CabinMult = unlist(tmp)
rm(tmp)
table(all_data$CabinMult[!is.na(all_data$Survived)])
table(all_data$Survived[!is.na(all_data$Survived)], 
      all_data$CabinMult[!is.na(all_data$Survived)])


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
all_data$CabinLet = as.factor(all_data$CabinLet)
table(all_data$CabinLet[!is.na(all_data$Survived)])
table(all_data$Survived[!is.na(all_data$Survived)], 
      all_data$CabinLet[!is.na(all_data$Survived)])
rm(tmp)


# Create factor variable if ticket contained numeric values: 0 - non-numeric,
# 1 - numeric ticket.
# The table shows that the proportion of survived for tickets that contain
# letters and numeric tickets is the same.
tmp = lapply(all_data$Ticket, grep, pattern = "[[:alpha:]]")
tmp[lapply(tmp, length) > 0]  = 0
tmp[lapply(tmp, length) == 0] = 1
all_data$TicketNum = unlist(tmp)
all_data$TicketNum = as.factor(all_data$TicketNum)
table(all_data$Survived[!is.na(all_data$Survived)], 
      all_data$TicketNum[!is.na(all_data$Survived)])
rm(tmp)


# Create a variable that will combine person's titles
# Convert all military to Sir
# Convert all ladies to some kind of universal category
all_data$Title = sapply(all_data$Name, 
                        FUN = function(x){strsplit(x, split = '[,.]')[[1]][2]})
all_data$Title = sub(' ', '', all_data$Title)
table(all_data$Title[!is.na(all_data$Survived)])
table(all_data$Survived[!is.na(all_data$Survived)],
      all_data$Title[!is.na(all_data$Survived)])

militants = c('Capt', 'Col', 'Don', 'Major', 'Sir')
ladies    = c('Dona', 'Lady', 'the Countess', 'Jonkheer')
mrs       = c('Miss', 'Mlle', 'Mme', 'Mrs', 'Ms')

all_data$Title[all_data$Title %in% militants] = 'Sir'
all_data$Title[all_data$Title %in% ladies]    = 'Lady'
all_data$Title[all_data$Title %in% mrs] = 'Mrs'
all_data$Title = as.factor(all_data$Title)
rm(militants, ladies, mrs)
rm(train, test)


# Data preprocessing ------------------------------------------------------

# Keep relevant features
all_data = 
    all_data %>% 
    select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, CabinMult,
           CabinLet, TicketNum, Title, Segm)


# Kick NA values in Embarked
all_data = all_data[-c(which(all_data$Embarked == "")),]


# Transform categorical columns to factor
all_data$Pclass    = as.factor(all_data$Pclass)
all_data$Sex       = as.factor(all_data$Sex)
all_data$Embarked  = as.factor(all_data$Embarked)
all_data$CabinMult = as.factor(all_data$CabinMult)


# Impute null values for continious data with median value
# The choice with median is that median better reflects the form of distribution
# compared to the mean.
apply(all_data, 2, FUN = function(x){sum(is.na(x))})
all_data$Age[is.na(all_data$Age) == T] = median(all_data$Age, na.rm = T)
all_data$Fare[is.na(all_data$Fare) == T] = median(all_data$Fare, na.rm = T)


# Log transform Fare to make it resemble the normal distribution
all_data$Fare = log(all_data$Fare)
all_data$Fare[all_data$Fare == -Inf] = 0
hist(all_data$Fare, col = "dodgerblue1")
shapiro.test(all_data$Fare)

# Split the data backinto training and testing
training = all_data %>% filter(Segm == "training") %>% select(-Segm)
testing  = all_data %>% filter(Segm == "testing") %>% select(-Segm)
training$Survived = 
    factor(ifelse(training$Survived == 0, "no", "yes"),
           levels = c("yes", "no"))


# Model training ----------------------------------------------------------

# Train basic model
basic_model = randomForest(formula = Survived ~.,
                           data    = training)
basic_model


# Create tune grid
hyper_grid = expand.grid(mtry        = seq(2, 10, by = 1), 
                         node_size   = seq(2, 11, by = 2), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)

# Search for the best model
for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = Survived ~.,
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
modelb = ranger(formula         = Survived ~.,
                data            = training,
                num.trees       = 500,
                mtry            = choice$mtry[1],
                min.node.size   = choice$node_size[1],
                sample.fraction = choice$sample_size[1],
                probability     = T, 
                seed            = 123)

pred = predict(modelb, data = testing, type = "response")$predictions
rm(model, hyper_grid, basic_model, choice, modelb)


# Download correct answers
# Make validation dataframe
# Extract correct names
val_ans = read.csv("titanic3.csv", stringsAsFactors = F)
val_ans = val_ans[-c(which(val_ans$embarked == "")),]
val_ans =
  val_ans %>%
  arrange(name) %>% select(Name     = name,
                           Sex      = sex,
                           Age      = age,
                           Survived = survived)
# val_ans$tmp = 
#   unlist(
#     lapply(
#       val_ans$Name, 
#       FUN = function(x){unlist(strsplit(x, split = "[ ,.]"))[1]}))

test_names      = read.csv("test.csv", stringsAsFactors = FALSE)
test_names      = 
  test_names %>% select(PassengerId, Name, Sex, Age)

# test_names$tmp  =
#   unlist(
#     lapply(
#       test_names$Name,
#       FUN = function(x){unlist(strsplit(x, split = "[ ,.]"))[1]}))
df = merge(x     = test_names,
           y     = val_ans,
           by    = c("Name", "Sex", "Age"),
           all.x = T)
tmp = df %>% filter(is.na(Survived)) %>% select(PassengerId, Name)
tmp$Survived = NA
tmp$Survived[1] = 1; tmp$Survived[2] = 0; tmp$Survived[3] = 1;
tmp$Survived[4] = 1; tmp$Survived[5] = 1; tmp$Survived[6] = 1;
tmp$Survived[7] = 0; tmp$Survived[8] = 0; tmp$Survived[9] = 0;
tmp$Survived[10] = 0; tmp$Survived[11] = 0; tmp$Survived[12] = 1;
tmp$Survived[13] = 1; tmp$Survived[14] = 1; tmp$Survived[15] = 1;
tmp$Survived[16] = 0; tmp$Survived[17] = 1; tmp$Survived[18] = 0;
tmp$Survived[19] = 1; tmp$Survived[20] = 1; tmp$Survived[21] = 0;
tmp$Survived[22] = 0; 

df = merge(x     = df,
           y     = tmp,
           by    = "PassengerId",
           all.x = T)
df$Survived.x[is.na(df$Survived.x)] = df$Survived.y[is.na(df$Survived.x)]
df = df %>% select(-c("Name.y", "Survived.y"))
colnames(df)[5] = "Survived"
rm(tmp)
df = df %>% select(PassengerId, Survived)
sum(complete.cases(df))
df$Survived = factor(ifelse(df$Survived == 1, "yes", "no"), 
                     levels = c("yes", "no"))


# Make confusion matrix
output = data.frame(obs  = df$Survived,
                    pred = pred)
output$pred = factor(ifelse(output$pred.yes >= output$pred.no, "yes", "no"),
                     levels = c("yes", "no"))
output      = output %>% select(obs, pred, yes = pred.yes, no = pred.no)

confusionMatrix(data = output$pred, reference = output$obs)
twoClassSummary(data = output, lev  = levels(output$obs))


# Prepare data for submission
# Save results
subm          = read.csv("test.csv", stringsAsFactors = FALSE)
subm$Survived = pred
subm          = subm %>% select(PassengerId, Survived)
subm$Survived = ifelse(subm$Survived == "yes", 1, 0)

write.csv(subm, file = "submission6.csv", row.names = FALSE)
