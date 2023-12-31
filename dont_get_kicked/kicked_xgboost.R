# load libraries
suppressMessages(library(tidyverse))
suppressMessages(library(tidymodels))
suppressMessages(library(vroom))
suppressMessages(library(corrplot))
suppressMessages(library(xgboost))
suppressMessages(library(embed)) # for target encoding
suppressMessages(library(themis)) # for balancing

train <- vroom('/kaggle/input/DontGetKicked/training.csv')
test <- vroom('/kaggle/input/DontGetKicked/test.csv')
idNumbers <- vroom('/kaggle/input/DontGetKicked/test.csv')


train[train == "NULL"] <- NA

test[test == "NULL"] <- NA


# predict and format function
predict_and_format <- function(workflow, newdata, filename){
  predictions <- predict(workflow, new_data = newdata, type = "prob")
  
  submission <- predictions %>% 
    mutate(RefId = test2$RefId) %>% 
    rename("IsBadBuy" = ".pred_1") %>% 
    select(3,2)
  
  vroom_write(submission, filename, delim = ',')
}


# convert characters to doubles
train$WheelTypeID <- as.double(train$WheelTypeID)
train$MMRCurrentAuctionAveragePrice <- as.double(train$MMRCurrentAuctionAveragePrice)
train$MMRCurrentAuctionCleanPrice <- as.double(train$MMRCurrentAuctionCleanPrice)
train$MMRCurrentRetailAveragePrice <- as.double(train$MMRCurrentRetailAveragePrice)
train$MMRCurrentRetailCleanPrice <- as.double(train$MMRCurrentRetailCleanPrice)


# unnecesary cols
IDs <- c('RefId', 'WheelTypeID', 'BYRNO')
categories <- c('PurchDate', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'VNZIP1', 'VNST')
high_corr <- c('MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailCleanPrice',
               'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice', 'VehYear')


drop_cols <- c(IDs, categories, high_corr)


# remove cols from train and test
train <- train[, !(names(train) %in% drop_cols)]
test <- test[, !(names(test) %in% drop_cols)]


## MODEL - stack naive bayes and random forest
# recipe for modeling
my_recipe <-  recipe(IsBadBuy ~ ., train) %>%
  step_novel(all_nominal_predictors(), -all_outcomes()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>% # target encoding
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold= 0.7) %>%
  step_zv() %>%
  step_normalize(all_numeric_predictors())

train$IsBadBuy <- as.factor(train$IsBadBuy)


xgboost_model <- boost_tree(trees = 100,
                       tree_depth = tune(), 
                       min_n = tune(),
                       loss_reduction = tune(),
                       mtry = tune(),
                       learn_rate = tune()) %>%
set_engine("xgboost") %>%
set_mode("classification")


## SET UP WORKFLOW
xgboost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(xgboost_model)

L <- 3
K <- 5

tuneGrid <- grid_regular(
  tree_depth(),
  min_n(),
  learn_rate(),
  loss_reduction(),
  mtry(range = c(1, ncol(train))),
  levels = L
)

folds <- vfold_cv(train, v = K, repeats = 1)

## RUN CV
CV_results <- xgboost_wf %>%
  tune_grid(resamples = folds,
            grid = tuneGrid,
            metrics = metric_set(roc_auc))

## FIND BEST TUNING PARAMETERS
xgboost_best_tune <- select_best(CV_results, "roc_auc")

final_xgboost_wf <- xgboost_wf %>%
  finalize_workflow(xgboost_best_tune) %>%
  fit(data = train)

## PREDICTIONS
predict_and_format(final_xgboost_wf, test, "submission.csv")