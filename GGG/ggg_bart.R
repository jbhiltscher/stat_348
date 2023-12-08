library(vroom)
library(dplyr)
library(recipes)
library(tidymodels)
library(dbarts)

##LOAD IN DATA
ggg_test <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/data/test.csv")

ggg_train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/train.csv")
ids <- ggg_test$id

## SET UP RECIPE
my_recipe <- recipe(formula =type~., data = ggg_train) %>%
  update_role(id, new_role = "id") %>%
  step_string2factor(color) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

## CREATE MODEL
initial_model <- rand_forest(trees = tune()) %>%
  set_engine("bart") %>%
  set_mode("classification")

## SET UP WORKFLOW
initial_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(initial_model)

K <- 2
L <- 3

tuneGrid <- grid_regular(tree_depth(),
                         trees(),
                         learn_rate(),levels = L)

folds <- vfold_cv(ggg_train, v = K, repeats = 1)

## RUN CV
CV_results <- initial_wf %>%
  tune_grid(resamples=folds,
            grid=tuneGrid,
            metrics=metric_set(roc_auc))

## FIND BEST TUNING PARAMETERS
best_tune <- CV_results %>%
  select_best("roc_auc")

## FINALIZE THE WORKFLOW WITH best_tune
final_wf <- initial_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = ggg_train)

## PREDICTIONS
ggg_preds <- predict(final_wf,
                     new_data = ggg_test,
                     type = "class")

ggg_preds <- ggg_preds %>%
  mutate(id = ids) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)


vroom_write(x=ggg_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/GGG/data/Test_Preds_lightgbm.csv", delim=",")

