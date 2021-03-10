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
# 


