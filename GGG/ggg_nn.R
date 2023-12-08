library(vroom)
library(dplyr)
library(recipes)
library(tidymodels)
library(keras)
library(tensorflow)

##LOAD IN DATA
ggg_test <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/test.csv")

ggg_train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/train.csv")

## SET UP RECIPE
my_recipe <- recipe(formula =type~., data = ggg_train) %>%
  update_role(id, new_role = "id") %>%
  step_string2factor(color) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

## CREATE MODEL
my_model <- mlp(hidden_units = tune(),
                epochs = 50,#or 100 or 250
                activation = "relu") %>%
  set_engine("keras", verbose=0) %>%
  set_mode("classification")

## SET UP WORKFLOW
initial_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

K <- 2
L <- 3
maxHiddenUnits <-50 

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, maxHiddenUnits)),
                            levels = L)

folds <- vfold_cv(ggg_train, v = K, repeats = 1)

## RUN CV
CV_results <- initial_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(roc_auc))

## FIND BEST TUNING PARAMETERS
best_tune <- CV_results %>%
  select_best("roc_auc")

## FINALIZE THE WORKFLOW WITH best_tune
final_wf <- nn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = ggg_train)

## PREDICTIONS
ggg_preds <- predict(final_wf,
                        new_data = ggg_test,
                        type = "class")


vroom_write(x=amazon_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/GGG/Test_Preds_keras.csv", delim=",")


