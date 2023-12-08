library(stacks)
library(tidymodels)
library(vroom)

## LOAD IN DATA AND CLEAN
bikeTrain <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/train.csv")
bikeTest <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/test.csv")

bikeTrain <- bikeTrain %>%
  select(-casual, - registered) %>%
  mutate(count = log(count)) # LOG Transformation

## CREATE RECIPE
bike_recipe <- recipe(count~., data=bikeTrain) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features="hour") %>% 
  step_date(datetime, features = "month") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## SET UP GRID OF TUNING VALUES
L <- 5
K <- 5

## Split data for CV
folds <- vfold_cv(bikeTrain, v = K, repeats = 1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = L) ## L^2 total tuning possibilities

## Run the CV
preg_models <- preg_wf %>%
tune_grid(resamples=folds,
          grid=preg_tuning_grid,
          metrics=metric_set(rmse, mae, rsq),
          control = untunedModel) # including the control grid in the tuning ensures you can6
# call on it later in the stacked model

## Create other resampling objects with different ML algorithms to include in a stacked model
evaluation_metrics <- metric_set(rmse)

lin_reg <- linear_reg() %>%
  set_engine("lm")
lin_reg_wf <- workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(bike_recipe)
lin_reg_model <- fit_resamples(
              lin_reg_wf,
              resamples = folds,
              control = tunedModel
              )
 
## Specify with model to include
my_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(lin_reg_model)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

## Use the stacked data to get a prediction
test_preds <- predict(stack_mod, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>%
  mutate(count=exp(count)) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) # needed for right format to Kaggle
## Write prediction file to CSV
vroom_write(x=test_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/TestPreds_stackedModel.csv", delim=",")



