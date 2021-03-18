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
# Correlation plot shows that there is a correlation between Age and other
# numeric variables -- with this it might be a good idea to predict age from
# other variables and do not use median.
apply(df_num, 2, FUN = function(x){sum(is.na(x))})
df_num = na.omit(df_num)
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


# Add family size
# Add travelling alone variable
all_data$fam_size = all_data$SibSp + all_data$Parch
hist(all_data$fam_size, col = "lightgrey")
all_data$FamSize = "small"
all_data$FamSize[all_data$fam_size > 1 & all_data$fam_size <= 3] = "medium"
all_data$FamSize[all_data$fam_size > 3 & all_data$fam_size <= 4] = "large"
all_data$FamSize[all_data$fam_size > 4] = "very large"
all_data$TravAlone = 0
all_data$TravAlone[all_data$fam_size == 0] = 1
all_data = all_data %>% select(-fam_size)
all_data$FamSize   = as.factor(all_data$FamSize)
all_data$TravAlone = as.factor(all_data$TravAlone)

all_data %>% group_by(FamSize) %>% summarise(count = n())
table(all_data$Survived[!is.na(all_data$Survived)], 
      all_data$FamSize[!is.na(all_data$Survived)])


# Survival rate differs across family size groups
# We can use the family size to judge the survival rate
all_data %>%
  filter(!is.na(Survived)) %>% 
  group_by(FamSize) %>% 
  summarise(count = n(), 
            surv  = sum(as.numeric(Survived), na.rm = T)) %>% 
  mutate(surv_rate = surv / count) %>% 
  ggplot2::ggplot() + geom_col(aes(x = FamSize, y = surv_rate), 
                               fill = "darkgreen")


# Impute null values for continious data with median value
# The choice with median is that median better reflects the form of distribution
# compared to the mean.
apply(all_data, 2, FUN = function(x){sum(is.na(x))})
all_data$Fare[is.na(all_data$Fare) == T] = median(all_data$Fare, na.rm = T)


# Impute empty values in Embarked with modal
all_data %>% group_by(Embarked) %>% summarise(count = n())
all_data$Embarked[all_data$Embarked == ""] = "S"


# Create has cabin variable
all_data$HasCabin = factor(ifelse(all_data$CabinMult == 0, "no", "yes"),
                           levels = c("yes", "no"))
table(all_data$Survived[!is.na(all_data$Survived)], 
      all_data$HasCabin[!is.na(all_data$Survived)])


# Predict Age and compare it to median subst case
# Choose one that has better accuracy

# # Example of Crammer's V computation using mock data
# # Use this for **2** categorical variables
# # TODO: DO NOT DELETE THIS -- use as a part of education in further reading
# tbl           = matrix(data = c(55, 45, 20, 30), 
#                        nrow = 2, ncol = 2, byrow = T)
# dimnames(tbl) = list(City=c('B', 'T'), Gender=c('M', 'F'))
# 
# chi2 = chisq.test(tbl, correct = F)
# 
# # The p-value is 0.08 > 0.05 so we can't reject the hypothesis of independence.
# # That means that the data here is independent.
# c(chi2$statistic, chi2$p.value) 
# 
# # Computation of Crammer's V shows the value of 0.14 which is small corr. The
# # higher the value of V the more the correlation is.
# sqrt(chi2$statistic / sum(tbl))
# 
# # Consider another data
# # Here the p-value is 0.72 and V-value is 0.03 -- so there is almost no
# # correlation between the variables
# tbl = matrix(data=c(51, 49, 24, 26), nrow=2, ncol=2, byrow=T)
# dimnames(tbl) = list(City=c('B', 'T'), Gender=c('M', 'F'))
# 
# chi2 = chisq.test(tbl, correct=F)
# c(chi2$statistic, chi2$p.value)
# sqrt(chi2$statistic / sum(tbl))
#
# rm(tbl, chi2)


# # Example of one-way ANOVA test -- use to find corr between Num and Cat vars
# # TODO: DO NOT DELETE THIS -- use as a part of education in further reading
# # H0: the amount of fat absorbed is equal for all 4 types of fat
# # P-value < 0.05 rejects H0
# t1 = c(164, 172, 168, 177, 156, 195)
# t2 = c(178, 191, 197, 182, 185, 177)
# t3 = c(175, 193, 178, 171, 163, 176)
# t4 = c(155, 166, 149, 164, 170, 168)
# 
# val = c(t1, t2, t3, t4)
# fac = gl(n=4, k=6, labels=c('type1', 'type2', 'type3', 'type4'))
# 
# oneway.test(val ~ fac, var.equal = T)
# aov1 = aov(val ~ fac)
# summary(aov1)


# # Another example of one-way ANOVA test to confirm understanding
# # There are four types of diet. The % of fat increases for diet types from 1-4.
# # H0: fat levels are equal for different diet types
# set.seed(seed)
# f1 = rnorm(n = 100, mean = 4,  sd = 0.8)
# f2 = rnorm(n = 100, mean = 6,  sd = 0.5)
# f3 = rnorm(n = 100, mean = 8,  sd = 1.0)
# f4 = rnorm(n = 100, mean = 10, sd = 0.7)
# 
# d1 = data.frame(fat = f1, diet = "d1")
# d2 = data.frame(fat = f2, diet = "d2")
# d3 = data.frame(fat = f3, diet = "d3")
# d4 = data.frame(fat = f4, diet = "d4")
# df = rbind(d1, d2, d3, d4)
# 
# rm(f1, f2, f3, f4)
# rm(d1, d2, d3, d4)
# 
# df$diet = as.factor(df$diet)
# boxplot(df$fat ~ df$diet)
# 
# # The p-value is < 0.05 -- we can reject the null hypothesis
# # This means that there is correlation between diet type and level of fat
# oneway.test(df$fat ~ df$diet, var.equal = T)

# Make correlation between Age and categorical variables
df_age = 
  all_data %>% 
  select(Age, Pclass, Sex, Embarked, CabinMult, 
         CabinLet, TicketNum, Title, HasCabin,
         TravAlone, FamSize)
df_age = df_age %>% filter(!is.na(Age))

# Test oneway ANOVA on Age ~ Pclass. Here the p-value < 0.05 rejects the null
# H0: passengers of different classes are equally distributed by age
oneway.test(df_age$Age ~ df_age$Pclass, var.equal = T)

# Oneway test for all variables shows that some variables correlate to Age
# The output of the test should be read as follows: the lower the squared error
# the higher is the p-value. In other words the decrease in SS confirms H0.
# This way corr vars are: Pclass, Sex, Title
stats::aov(Age ~., data = df_age)

# Confirm the computation above using ANOVA
# ANOVA shows that p-values < 0.05 are true for: Pclass, Sex, Embarked, Title
anova(lm(Age ~., data = df_age))


# Make regression model to predict Age of a passenger in order to substitue 
# NA values. Choose variables correlating with
# Age ~ SibSp + Pclass + Sex + Embarked + Title
df_age = all_data %>% select(Age, Pclass, Sex, Title, CabinLet)

df_ag = na.omit(df_age)

set.seed(seed)
age_intrain = createDataPartition(df_ag$Age, p = 0.8, list = F)
age_train   = df_ag[age_intrain, ]
age_test    = df_ag[-age_intrain,]

agesh_grid = expand.grid(mtry        = seq(1, 3, by = 1), 
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
all_age_pred = predict(modela, data = df_age, type = "response")$predictions
all_age_pred = round(all_age_pred, 0)

all_data$age_temp = all_age_pred
all_data$Age[is.na(all_data$Age)] = all_data$age_temp[is.na(all_data$Age)]
all_data = all_data %>% select(-age_temp)

# Make age groups
# Make a plot to see the correlation between age and survival rate
# The plot shows that survival rates differ by 10 year age groups
hist(all_data$Age[all_data$Survived == 1], col = "dodgerblue1")
surv_by_age = 
  all_data %>% 
  na.omit() %>% 
  select(Age, Survived)
surv_by_age$RoundAge = round(surv_by_age$Age, digits = -1)
  
surv_by_age %>% 
  group_by(RoundAge) %>% 
  summarise(count = n(), surv  = sum(as.numeric(Survived), na.rm = T)) %>% 
  mutate(surv_rate = surv / count) %>%
  ggplot2::ggplot() + geom_point(aes(x = RoundAge, 
                                     y = surv_rate, 
                                     col = RoundAge),
                                 alpha = 3/4)

all_data$AgeGroup = "kids"
all_data$AgeGroup[all_data$Age >= 10 & all_data$Age <= 30] = "young"
all_data$AgeGroup[all_data$Age > 30 & all_data$Age <= 60]  = "mid"
all_data$AgeGroup[all_data$Age > 60]                       = "elder"




# Make survival by age group
all_data %>%
  filter(!is.na(Survived)) %>% 
  group_by(AgeGroup) %>% 
  summarise(count = n(), 
            surv  = sum(as.numeric(Survived), na.rm = T)) %>% 
  mutate(surv_rate = surv / count) %>% 
  ggplot2::ggplot() + geom_col(aes(x = AgeGroup, y = surv_rate), 
                               fill = "darkgreen")
all_data$AgeGroup = as.factor(all_data$AgeGroup)

rm(df_ag, df_age, model, modela, surv_by_age, age_choice, age_intrain,
   age_test, age_train, agesh_grid)
rm(age_median, age_pred, all_age_pred, err_median, error, i, rmse_med,
   rmse_out)

# Make Fare group
hist(all_data$Fare, col = "lightgreen")
hist(all_data$Fare[all_data$Fare < 50], col = "lightgreen")

all_data$TempFare = "cheap"
all_data$TempFare[all_data$Fare > 10 & all_data$Fare <= 50] = "medium"
all_data$TempFare[all_data$Fare > 50] = "expensive"

all_data %>% 
  na.omit() %>% 
  group_by(TempFare) %>% 
  summarise(count = n(), 
            surv  = sum(as.numeric(Survived), na.rm = T)) %>% 
  mutate(surv_rate = surv / count) %>% 
  ggplot2::ggplot() + geom_col(aes(x = TempFare, y = surv_rate), 
                               fill = "darkgreen")
all_data$TempFare = as.factor(all_data$TempFare)


# Data preprocessing ------------------------------------------------------

# Keep relevant features
all_data = 
    all_data %>% 
    select(Survived, 
           Pclass, 
           Sex,
           Embarked,
           Segm,
           CabinMult,
           CabinLet,
           TicketNum,
           Title,
           FamSize,
           TravAlone,
           HasCabin,
           AgeGroup,
           TempFare)


# Transform categorical columns to factor
all_data$Pclass    = as.factor(all_data$Pclass)
all_data$Sex       = as.factor(all_data$Sex)
all_data$Embarked  = as.factor(all_data$Embarked)
all_data$CabinMult = as.factor(all_data$CabinMult)


# TODO: check for correlated variables
# TODO: make balanced dataset for training


# Split the data backinto training and testing
training = all_data %>% filter(Segm == "training") %>% select(-Segm)
testing  = all_data %>% filter(Segm == "testing") %>% select(-Segm)
training$Survived = 
    factor(ifelse(training$Survived == 0, "no", "yes"),
           levels = c("yes", "no"))


# Model training ----------------------------------------------------------

# Train basic model
basic_model = randomForest(formula = Survived ~ Sex + Pclass + Embarked + AgeGroup + HasCabin,
                           data    = training)
basic_model


# Create tune grid
hyper_grid = expand.grid(mtry        = seq(2, 3, by = 1), 
                         node_size   = seq(2, 15, by = 2), 
                         sample_size = c(0.55, 0.632, 0.70, 0.80), 
                         OOB_RMSE    = 0)

# Search for the best model
for (i in 1:nrow(hyper_grid)){
    
    model = ranger(formula         = Survived ~ Sex + Pclass + AgeGroup,
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


test_names      = read.csv("test.csv", stringsAsFactors = FALSE)
test_names      = 
  test_names %>% select(PassengerId, Name, Sex, Age)


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
pred          = data.frame(pred)
pred$Survived = 0
pred$Survived[pred$yes >= pred$no] = 1
subm$Survived = pred$Survived
subm          = subm %>% select(PassengerId, Survived)

write.csv(subm, file = "submission9.csv", row.names = FALSE)
