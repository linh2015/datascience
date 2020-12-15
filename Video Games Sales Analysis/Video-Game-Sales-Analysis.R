################################################################################
################################################################################
# title: "Video Game Sales Analysis"
# author: "Linh Hua"
# date: "Decemeber 12, 2020"
################################################################################

################################################################################
# Generate the vgsales (video game sales) data
################################################################################

# Install libraries if they have not been installed yet
if(!require(tidyverse)) install.packages("tidyverse", 
                                          repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                          repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", 
                                          repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", 
                                          repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", 
                                          repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", 
                                          repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", 
                                     repos = "http://cran.us.r-project.org")

# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(class)
library(randomForest)
library(knitr)
library(kableExtra)

# Video games sales data set:
# https://www.kaggle.com/gregorut/videogamesales
# Manually downloaded and included the vgsales.csv file
# Use the below code to import programmically or use R Studio to import the data
filename <- "vgsales.csv"
dir <- system.file("extdata", package = "dslabs") 
fullpath <- file.path(dir, filename)
file.copy(fullpath, "vgsales.csv")

vgsales <- read_csv(filename)

################################################################################
# Clean the data
################################################################################

# Remove rows with N/A values in the Year column
vgsales <- vgsales %>% filter(Year != "N/A")

# Remove rows with N/A values in the Publisher column
vgsales <- vgsales %>% filter(Publisher != "N/A")

##################### Data Visualization and Exploration #######################

# View the structure of the data set
str(vgsales)

# View data summary
summary(vgsales)

# View number of rows
nrow(vgsales)

# View number of columns
ncol(vgsales)

# View different game platforms 
platforms <- unique(vgsales$Platform)
platforms

# View different game platforms 
publishers <- unique(vgsales$Publisher)
publishers 

# Calculate the mean of game sales in North America
avg_NA_sales <- mean(vgsales$NA_Sales)
avg_NA_sales

# Calculate the standard deviation of game sales in North America
sd_NA_sales <- sd(vgsales$NA_Sales)
sd_NA_sales

# Calculate the mean of global game sales 
avg_G_sales <- mean(vgsales$Global_Sales)
avg_G_sales

# Calculate the standard deviation of global game sales 
sd_G_sales <- sd(vgsales$Global_Sales)
sd_G_sales

# View bottom game sales in North America
worst_game_sales <- which.min(vgsales$NA_Sales)
vgsales$Name[worst_game_sales]
min(vgsales$NA_Sales)

# View top game sales in North America
best_game_sales <- which.max(vgsales$NA_Sales)
vgsales$Name[best_game_sales]
max(vgsales$NA_Sales)

# View bottom game sales Globally
worst_game_sales_g <- which.min(vgsales$Global_Sales)
vgsales$Name[worst_game_sales_g]
min(vgsales$Global_Sales)

# View top game sales Globally
best_game_sales_g <- which.max(vgsales$Global_Sales)
vgsales$Name[best_game_sales_g]
max(vgsales$Global_Sales)

# Bar plot: Games release by year counts
vgsales %>% 
  ggplot(aes(Year)) + 
  labs(x = "Game Release Year") +
  geom_bar(fill = "blue", col = "black") +
  coord_flip()

# Bar plot: Platform counts
vgsales %>% 
  ggplot(aes(Platform)) + 
  labs(x = "Platforms") +
  geom_bar() +
  coord_flip()

# Bar plot: Genre counts
vgsales %>% 
  ggplot(aes(Genre)) + 
  labs(x = "Genres") +
  geom_bar(fill = "green", col = "black") +
  coord_flip()

# Scatter plot: North America sales by Year
vgsales %>% 
  ggplot() +
  geom_point(aes(x = Year, y = NA_Sales)) +
  labs(x = "Years", y = "North America Sales") +
  theme(axis.text.x = element_text(angle = 90))

# Scatter plot: Global sales by Year
vgsales %>% 
  ggplot() +
  geom_point(aes(x = Year, y = Global_Sales)) +
  labs(x = "Years", y = "Global Sales") +
  theme(axis.text.x = element_text(angle = 90))

################################## Data Modeling ###############################

##### kNN Model: #####

## Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

## Selecting our predictors 
vgsales_subset <- vgsales %>%
  select(NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales)

## Creating the normalized subset
vgsales_subset_n <- as.data.frame(lapply(vgsales_subset[,1:5], normalize))

## Split data into a training set and a test set by randomly selection
set.seed(100, sample.kind = "Rounding")
d <- sample(1:nrow(vgsales_subset_n),size=nrow(vgsales_subset_n)*0.7,
            replace = FALSE) 

train_set <- vgsales_subset[d,] # 70% training data
test_set <- vgsales_subset[-d,] # remaining 30% test data

## Creating separate data frame for 'status' feature which is our target.
train_labels <- vgsales_subset[d,1]
test_labels <- vgsales_subset[-d,1]

## Find the number of observations
NROW(train_labels) 

## Square root of 11,403 is 106.78 and so we will create 2 models. 
## One with ‘K’ value as 106 and the other model with a ‘K’ value as 107.
knn_106 <- knn(train=train_set, test=test_set, cl=train_labels$NA_Sales, k=106)
knn_107 <- knn(train=train_set, test=test_set, cl=train_labels$NA_Sales, k=107)

## Calculate the proportion of correct classification for k = 106, 107
ACC_106 <- 100 * sum(test_labels$NA_Sales == knn_106)/NROW(test_labels$NA_Sales)
ACC_107 <- 100 * sum(test_labels$NA_Sales == knn_107)/NROW(test_labels$NA_Sales)
ACC_106
# [1] 54.31669
ACC_107
# [1] 54.29624

## Loop that calculates the accuracy of the kNN model
i=1
k_optm=1
for (i in 1:108){
  knn_mod <- knn(train=train_set, test=test_set, cl=train_labels$NA_Sales, k=i)
  k_optm[i] <- 100 * sum(test_labels$NA_Sales==knn_mod)/NROW(test_labels$NA_Sales)
  k=i
  cat(k,'=',k_optm[i],'')
}

## Accuracy plot
plot(k_optm, type="b", xlab="K- Value",ylab="Accuracy level")

#####  Naive Bayes Model: ##### 

## Selecting our predictors 
vgsales_subset <- vgsales %>%
  select(Genre, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales)

# Translate the Genre column to Action or Not Action and save it to Genre_Spec
vgsales_subset <- vgsales_subset %>%
  mutate(Genre_Spec = ifelse(Genre == "Action", "Action", "Not Action"))

## Split data into train and test sets
set.seed(1995)
test_index <- createDataPartition(vgsales_subset, times = 1, p = 0.5, list = FALSE)
train_set <- vgsales_subset %>% slice(-test_index) # 50% data
test_set <- vgsales_subset %>% slice(test_index)   # 50% remaining data

params <- train_set %>% 
  group_by(Genre_Spec) %>% 
  summarize(avg = mean(NA_Sales), sd = sd(NA_Sales))

pi <- train_set %>% summarize(pi=mean(Genre_Spec=="Action")) %>% pull(pi)
pi

x <- test_set$NA_Sales

f0 <- dnorm(x, params$avg[2], params$sd[2])
f1 <- dnorm(x, params$avg[1], params$sd[1])

p_hat_bayes <- f1*pi / (f1*pi + f0*(1 - pi))

y_hat_bayes <- ifelse(p_hat_bayes > 0.5, "Action", "Not Action")
sensitivity(data = factor(y_hat_bayes), reference = factor(test_set$Genre_Spec))
# [1] 0
specificity(data = factor(y_hat_bayes), reference = factor(test_set$Genre_Spec))
# [1] 1

## Now we do the same as above but unbiased
p_hat_bayes_unbiased <- f1 * 0.5 / (f1 * 0.5 + f0 * (1 - 0.5)) 
y_hat_bayes_unbiased <- ifelse(p_hat_bayes_unbiased > 0.5, "Action", "Not Action")

sensitivity(factor(y_hat_bayes_unbiased), factor(test_set$Genre_Spec))
# [1] 0.9342508
specificity(factor(y_hat_bayes_unbiased), factor(test_set$Genre_Spec))
# [1] 0.0555767

# Plotting the new rule at 1
qplot(x, p_hat_bayes_unbiased, geom = "line") + 
  geom_hline(yintercept = 0.5, lty = 2) + 
  geom_vline(xintercept = 1, lty = 2)

#####  Random Forests Model: ##### 

## Translate the values of Genre into Genre_No as follows:
train_set <- train_set %>% 
  mutate(Genre_No = case_when(
    Genre == "Strategy" ~ 1,
    Genre == "Sports" ~ 2,
    Genre == "Simulation" ~ 3,
    Genre == "Shooter" ~ 4,
    Genre == "Role-Playing" ~ 5,
    Genre == "Racing" ~ 6,
    Genre == "Puzzle" ~ 7,
    Genre == "Platform" ~ 8,
    Genre == "Misc" ~ 9,
    Genre == "Fighting" ~ 10,
    Genre == "Adventure" ~ 11,
    Genre == "Action" ~ 12,
  ))

test_set <- test_set %>% 
  mutate(Genre_No = case_when(
    Genre == "Strategy" ~ 1,
    Genre == "Sports" ~ 2,
    Genre == "Simulation" ~ 3,
    Genre == "Shooter" ~ 4,
    Genre == "Role-Playing" ~ 5,
    Genre == "Racing" ~ 6,
    Genre == "Puzzle" ~ 7,
    Genre == "Platform" ~ 8,
    Genre == "Misc" ~ 9,
    Genre == "Fighting" ~ 10,
    Genre == "Adventure" ~ 11,
    Genre == "Action" ~ 12,
  ))

## Summary of data in train and test sets
summary(train_set)
summary(test_set)

## Create a Random Forest model with default parameters
model1 <- randomForest(Genre_No ~ ., data = train_set, importance = TRUE)
model1

model2 <- randomForest(Genre_No ~ ., data = train_set, ntree = 500, mtry = 6, 
                       importance = TRUE)
model2

# Predicting on train set
predTrain <- predict(model2, train_set, type = "class")

# Checking classification accuracy
table(predTrain, train_set$Genre_No) 

# Predicting on test set
predValid <- predict(model2, test_set, type = "class")

# Checking classification accuracy
mean(predValid == test_set$Genre_No)                    
table(predValid, test_set$Genre_No)

# To check important variables
importance(model2)        
varImpPlot(model2)  

##################################### Results ##################################

# Results Table(s)
results_discovery <- tibble(Description = "Data rows", Value = "16,291")
results_discovery <- results_discovery %>% 
  add_row(Description = "Variables", Value = "11")
results_discovery <- results_discovery %>% 
  add_row(Description = "Number of places games were sold", Value = "5")
results_discovery <- results_discovery %>% 
  add_row(Description = "Names of places games were sold", 
          Value = "North America, Europe, Japan, Other and Global ")
results_discovery <- results_discovery %>% 
  add_row(Description = "Number of Publishers", Value = "576")
results_discovery <- results_discovery %>% 
  add_row(Description = "Number of Platforms", Value = "31")
results_discovery <- results_discovery %>% 
  add_row(Description = "Average video games sales (North America)", 
          Value = "0.265 million")
results_discovery <- results_discovery %>% 
  add_row(Description = "Standard deviation (North America)", Value = "0.822")
results_discovery <- results_discovery %>% 
  add_row(Description = "Average video games sales (Globally)", 
          Value = "0.540 million")
results_discovery <- results_discovery %>% 
  add_row(Description = "Standard deviation (Globally)", Value = "1.567")
results_discovery <- results_discovery %>% 
  add_row(Description = "Top 3 game platforms", 
          Value = "Nintendo DS, PlayStation 2 and PlayStation 3")
results_discovery <- results_discovery %>% 
  add_row(Description = "Top 3 years to release the most games", 
          Value = "2008, 2009 and 2010")
results_discovery <- results_discovery %>% 
  add_row(Description = "Top 3 game genres", 
          Value = "Action, Sports and Miscellaneous")
results_discovery <- results_discovery %>% 
  add_row(Description = "Worst seller (North America)", 
          Value = "Monster Hunter Freedom 3 ($0)")
results_discovery <- results_discovery %>% 
  add_row(Description = "Worst seller (Globally)", 
          Value = "Turok (0.01 million)")
results_discovery <- results_discovery %>% 
  add_row(Description = "Best seller (North America)", 
          Value = "Wii Sports (41.49 million)")
results_discovery <- results_discovery %>% 
  add_row(Description = "Best seller (Globally)", 
          Value = "Wii Sports  (82.74 million) ")

kable(results_discovery) %>%
  kable_styling(latex_options = "striped")

# kNN Results
results_kNN <- tibble(Description = "ACC_106", Result = "54.50082")
results_kNN <- results_kNN %>% 
  add_row(Description = "ACC_106", Result = "54.41899")

kable(results_kNN) %>%
  kable_styling(latex_options = "striped")

# Naive Bayes Results
results_NB <- tibble(Description = "Sensitivity factor (y_hat_bayes_unbiased)", 
                     Result = "0.9342508")
results_NB <- results_NB %>% 
  add_row(Description = "Specificity factor (y_hat_bayes_unbiased)", 
                     Result = "0.0555767")

kable(results_NB) %>%
  kable_styling(latex_options = "striped")

# Random Forests Results
results_RF <- tibble(Description = "Model 1: Mean of squared residuals", 
                     Result = "0.0776066")
results_RF <- results_RF %>% 
  add_row(Description = "Model 1: % var explained", Result = "99.45")
results_RF <- results_RF %>% 
  add_row(Description = "Model 2: Mean of squared residuals", 
          Result = "0.0002617268")
results_RF <- results_RF %>% 
  add_row(Description = "Model 2: % var explained", Result = "100")
results_RF <- results_RF %>% 
  add_row(Description = "Importance: Genre", Result = "191.67")
results_RF <- results_RF %>% 
  add_row(Description = "Importance: NA_Sales", Result = "20.85")
results_RF <- results_RF %>% 
  add_row(Description = "Importance: Genre_Spec", Result = "10.19")

kable(results_RF) %>%
  kable_styling(latex_options = "striped")


