library(vroom)
library(dplyr)
library(recipes)
library(tidymodels)
library(lightgbm)
library(bonsai)
library(embed)

##LOAD IN DATA
ggg_test <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/data/test.csv")

ggg_train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/train.csv")
ids <- ggg_test$id
ggg_train$type <- as.factor(ggg_train$type)

library(dplyr)

ggg_train <- ggg_train %>%
  mutate(
    bone_flesh = bone_length * rotting_flesh,
    bone_hair = bone_length * hair_length,
    bone_soul = bone_length * has_soul,
    flesh_hair = rotting_flesh * hair_length,
    flesh_soul = rotting_flesh * has_soul,
    hair_soul = hair_length * has_soul
  ) %>%
  select(-c(color))

ggg_test <- ggg_test %>%
  mutate(
    bone_flesh = bone_length * rotting_flesh,
    bone_hair = bone_length * hair_length,
    bone_soul = bone_length * has_soul,
    flesh_hair = rotting_flesh * hair_length,
    flesh_soul = rotting_flesh * has_soul,
    hair_soul = hair_length * has_soul
  ) %>%
  select(-c(color))

## SET UP RECIPE
my_recipe <- recipe(formula =type ~id+ bone_length + rotting_flesh + hair_length + has_soul + hair_soul + bone_flesh + bone_hair + 
                      bone_soul + flesh_hair + flesh_soul, data = ggg_train) %>%
  update_role(id, new_role = "id") %>%
  step_range(all_numeric_predictors(), min=0, max=1)

## CREATE MODEL
initial_model <- boost_tree(tree_depth = tune(),
                       trees = tune(),
                       learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

## SET UP WORKFLOW
initial_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(initial_model)

K <- 3
L <- 3

tuneGrid <- grid_regular(tree_depth(),
                         trees(),
                         learn_rate(),levels = L)

folds <- vfold_cv(ggg_train, v = K, repeats = 1)

## RUN CV
CV_results <- initial_wf %>%
  tune_grid(resamples=folds,
            grid=tuneGrid,
            metrics=metric_set(accuracy))


## FIND BEST TUNING PARAMETERS
best_tune <- CV_results %>%
  select_best("accuracy")

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

