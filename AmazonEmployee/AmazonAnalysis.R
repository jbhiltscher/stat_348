library(tidymodels)
library(embed)
library(vroom)
library(ggplot2)

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
