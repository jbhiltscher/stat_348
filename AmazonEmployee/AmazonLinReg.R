library(tidymodels)
library(vroom)

## LOAD IN DATA
amazonTrain <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/AmazonEmployeeAccess/train.csv")
amazonTest <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/AmazonEmployeeAccess/test.csv")

amazonTrain <- amazonTrain %>% mutate(ACTION = as.factor(ACTION))

## CREATE RECIPE
my_recipe <- recipe(ACTION ~ ., data = amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold=0.01) %>%
  step_dummy(all_nominal_predictors()) # DUMMY ENCODING
# step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var))  # TARGET ENCODING

## APPLY THE RECIPE
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazonTrain)

## SET UP MODEL
my_mod <- logistic_reg() %>%
  set_engine("glm")

## SET UP WORKFLOW
amazon_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazonTrain)

## PREDICTIONS
amazon_preds <- predict(amazon_wf,
                        new_data = amazonTest,
                        type = "prob")

amazon_preds <- amazon_preds %>%
  mutate(id = c(1:nrow(amazon_preds))) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(x=amazon_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/AmazonEmployeeAccess/Test_Preds_Lin_Reg.csv", delim=",")
