library(recipes)
library(embed)
library(tidymodels)
library(glmnet)
library(vroom)
library(themis)
library(ranger)

## LOAD IN DATA
amazonTrain <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/AmazonEmployeeAccess/test.csv")

amazonTrain <- amazonTrain %>% mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # TARGET ENCODING
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes())

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

amazon_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazonTrain)

## GRID OF VALUES TO TUNE OVER
L <- 3
K <- 5
tuning_grid <- grid_regular(mtry(range(c(1,ncol(amazonTrain)))),
                            min_n(),
                            levels = L)

## SPLIT DATA FOR CV
folds <- vfold_cv(amazonTrain, v = K, repeats = 1)

CV_results <- amazon_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## FIND BEST TUNING PARAMETERS
best_tune <- CV_results %>%
  select_best("roc_auc")

## FINALIZE THE WORKFLOW WITH best_tune
final_wf <- amazon_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = amazonTrain)

best_tune

## PREDICTIONS
amazon_preds <- predict(final_wf,
                        new_data = amazonTest,
                        type = "prob")

amazon_preds <- amazon_preds %>%
  mutate(id = c(1:nrow(amazon_preds))) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=amazon_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/AmazonEmployeeAccess/Test_Preds_RandForest.csv", delim=",")
