library(vroom)
library(dplyr)
library(tidymodels)
library(ranger)
library(randomForest)
library(embed)
library(themis)


# Load Data
raw_train <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/training.csv')
raw_test <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/test.csv')

# Clean Datasets
train <- raw_train %>%
  select(-c(PRIMEUNIT, AUCGUART, VehYear, BYRNO, WheelTypeID, VNST, VNZIP1, PurchDate, Model, SubModel, Trim)) # Drop Columns
test <- raw_test %>%
  select(-c(PRIMEUNIT, AUCGUART, VehYear, BYRNO, WheelTypeID, VNST, VNZIP1, PurchDate, Model, SubModel, Trim)) # Drop Columns

mmr_columns <- grep("^MMR", names(train), value = TRUE)

my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  update_role(RefId, new_role = "id") %>%
  step_naomit(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_mean(all_predictors(), columns = mmr_columns) %>%
  step_smote(all_outcomes())


# Set up model
my_model <- rand_forest(mtry = 100,
                      min_n = 6,
                      trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Set up workflow
kicked_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model) %>%
  fit(data = train)


## PREDICTIONS
kicked_preds <- predict(kicked_wf,
                        new_data = test,
                        type = "class")
