# load libraries
suppressMessages(library(tidyverse))
suppressMessages(library(tidymodels))
suppressMessages(library(vroom))
suppressMessages(library(corrplot))
suppressMessages(library(parsnip)) # BART
suppressMessages(library(embed)) # for target encoding
suppressMessages(library(themis)) # for balancing
library(stacks)

train <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/training.csv')
test <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/test.csv')
idNumbers <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/test.csv')


train[train == "NULL"] <- NA

test[test == "NULL"] <- NA


# predict and format function
predict_and_format <- function(workflow, newdata, filename){
  predictions <- predict(workflow, new_data = newdata, type = "prob")
  
  submission <- predictions %>% 
    mutate(RefId = idNumbers$RefId) %>% 
    rename("IsBadBuy" = ".pred_1") %>% 
    select(3,2)
  
  vroom_write(submission, filename, delim = ',')
}

## EDA
# Look at column data types
str(train)


# convert characters to doubles
train$WheelTypeID <- as.double(train$WheelTypeID)
train$MMRCurrentAuctionAveragePrice <- as.double(train$MMRCurrentAuctionAveragePrice)
train$MMRCurrentAuctionCleanPrice <- as.double(train$MMRCurrentAuctionCleanPrice)
train$MMRCurrentRetailAveragePrice <- as.double(train$MMRCurrentRetailAveragePrice)
train$MMRCurrentRetailCleanPrice <- as.double(train$MMRCurrentRetailCleanPrice)

# select all numeric columns for correlation plot
numeric <- train %>%
  select(IsBadBuy, VehYear, VehicleAge, WheelTypeID, VehOdo, MMRAcquisitionAuctionAveragePrice,
         MMRAcquisitionAuctionCleanPrice, MMRAcquisitionRetailAveragePrice, MMRAcquisitonRetailCleanPrice,
         MMRCurrentAuctionAveragePrice, MMRCurrentAuctionCleanPrice, MMRCurrentRetailAveragePrice, MMRCurrentRetailCleanPrice,
         BYRNO, VNZIP1, VehBCost, IsOnlineSale, WarrantyCost) %>% 
  na.omit()



# unnecesary cols
IDs <- c('RefId', 'WheelTypeID', 'BYRNO')
categories <- c('PurchDate', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'VNZIP1', 'VNST')
high_corr <- c('MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailCleanPrice',
               'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice', 'VehYear')

drop_cols <- c(IDs, categories, high_corr)

# remove cols from train and test
train <- train[, !(names(train) %in% drop_cols)]
test <- test[, !(names(test) %in% drop_cols)]

# MISSING VALUES
columns_with_missing_values <- colnames(train)[apply(is.na(train), 2, any)]
columns_with_missing_values


# replace missing numeric values with the median
train$MMRAcquisitionAuctionAveragePrice[is.na(train$MMRAcquisitionAuctionAveragePrice)] <- median(train$MMRAcquisitionAuctionAveragePrice,na.rm = TRUE)
train$MMRAcquisitionRetailAveragePrice[is.na(train$MMRAcquisitionRetailAveragePrice)] <- median(train$MMRAcquisitionRetailAveragePrice,na.rm = TRUE)
train$MMRCurrentAuctionAveragePrice[is.na(train$MMRCurrentAuctionAveragePrice)] <- median(train$MMRCurrentAuctionAveragePrice,na.rm = TRUE)
train$MMRCurrentRetailAveragePrice[is.na(train$MMRCurrentRetailAveragePrice)] <- median(train$MMRCurrentRetailAveragePrice,na.rm = TRUE)

test$MMRAcquisitionAuctionAveragePrice[is.na(test$MMRAcquisitionAuctionAveragePrice)] <- median(test$MMRAcquisitionAuctionAveragePrice,na.rm = TRUE)
test$MMRAcquisitionRetailAveragePrice[is.na(test$MMRAcquisitionRetailAveragePrice)] <- median(test$MMRAcquisitionRetailAveragePrice,na.rm = TRUE)
test$MMRCurrentAuctionAveragePrice[is.na(test$MMRCurrentAuctionAveragePrice)] <- median(test$MMRCurrentAuctionAveragePrice,na.rm = TRUE)
test$MMRCurrentRetailAveragePrice[is.na(test$MMRCurrentRetailAveragePrice)] <- median(test$MMRCurrentRetailAveragePrice,na.rm = TRUE)

# replace missing character values with unknown category
missing <- c('Transmission', 'WheelType', 'Nationality', 'Size',
             'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART')

for (i in missing) {
  train[[i]] <- ifelse(is.na(train[[i]]), 'Unknown', train[[i]])
  test[[i]] <- ifelse(is.na(test[[i]]), 'Unknown', test[[i]])
}

columns_with_missing_values <- colnames(train)[apply(is.na(train), 2, any)]
columns_with_missing_values # none!


## MODEL - stack naive bayes and random forest
# recipe for modeling
my_recipe <-  recipe(IsBadBuy ~ ., train) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) # target encoding

train$IsBadBuy <- as.factor(train$IsBadBuy)



# BART -----------------------------------------------------
# ... (previous code)

# BART -----------------------------------------------------
bart_model <- parsnip::bart(tree = tune()) %>%
  set_engine('dbarts') %>%
  set_mode("classification")

## SET UP WORKFLOW
bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

L <- 3
K <- 5

# Specify the grid for tuning parameters
tuneGrid <- grid_regular(
  trees(),
  levels = L
)

folds <- vfold_cv(train, v = K, repeats = 1)

## RUN CV
CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuneGrid,
            metrics = metric_set(accuracy))

## FIND BEST TUNING PARAMETERS
bart_best_tune <- select_best(CV_results, "accuracy")

final_bart_wf <- bart_wf %>%
  finalize_workflow(bart_best_tune) %>%
  fit(data = train)

## PREDICTIONS
predict_and_format(final_bart_wf, test, "C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/xgboost_preds.csv")
