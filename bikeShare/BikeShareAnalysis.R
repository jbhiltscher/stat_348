library(readr)
library(tidymodels)
library(poissonreg)
library(vroom)

bikeTrain <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/train.csv")
bikeTest <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/test.csv")

# If weather == 4 then change it to 3, if not keep same value
# From Kaggle: 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
clean_bikeTrain <- bikeTrain %>% mutate(weather = ifelse(weather==4,3,weather)) %>%
  mutate(count = log(count)) %>%
  select(-casual,-registered)


my_recipe <- recipe(count~., data = clean_bikeTrain) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_nzv()

prepped_recipe <- prep(my_recipe)

new_ <- bake(prepped_recipe, new_data = clean_bikeTrain)

## Set up model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>%
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
L <- 5
K <- 10

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = L) ## L^2 total possibilites

## Split data for CV
folds <- vfold_cv(new_trainSet, v = K, repeats = 1)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resample=folds,
            grid = tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) ## Or leave metrics NULL

## Plot Results
collect_metrics(CV_results) %>%
  filter(.metric == 'rmse') %>%
  ggplot(data=., aes(x=penalty, y = mean, color = factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

## Finalize the Workflow & fit it
final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bikeTrain)

## Predict
## Get Predictions for test set AND format for Kaggle
test_preds <- predict(final_wf, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>%
  mutate(count=exp(count)) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) # needed for right format to Kaggle

vroom_write(x=test_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/TestPreds_tuningModels.csv", delim=",")

