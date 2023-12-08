library(dbarts)
library(vroom)
library(tidymodels)

## LOAD IN DATA AND CLEAN
bikeTrain <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/train.csv")
bikeTest <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/test.csv")

num_trees <- 1000
L <- 5
K <- 5


## SPLIT DATA
bikeTrain_registered <- bikeTrain %>%
  select(-casual, -count) %>%
  mutate(registered = ifelse(registered > 0, log(registered), registered))

## CREATE RECIPE
registered_bike_recipe <- recipe(registered ~ ., data = bikeTrain_registered) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features="hour") %>% 
  step_date(datetime, features = "month") %>%
  step_date(datetime, features = "year") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## CREATE MODEL (FOREST)
registered_mod <- bart(trees = num_trees) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

## CREATE A WORKFLOW WITH MODEL AND RECIPE
registered_wf <- workflow() %>%
  add_recipe(registered_bike_recipe) %>%
  add_model(registered_mod)

## SET UP GRID OF TUNING VALUES


registered_tuning_grid <- grid_regular(mtry(range(c(1,ncol(bikeTrain_registered)))),
                                       min_n(), 
                                       levels = L) ## L^2 total possibilites

## SET UP K-FOLD CV
registered_folds <- vfold_cv(bikeTrain_registered, v = K, repeats = 1)

## FIND BEST TUNING PARAMETERS
registered_CV_results <- registered_wf %>%
  tune_grid(resample = registered_folds,
            grid = registered_tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) ## Or leave metrics NULL

registered_bestTune <- registered_CV_results %>%
  select_best("rmse")

## FINALIZE WORKFLOW AND PREDICT
final_registered_wf <- registered_wf %>%
  finalize_workflow(registered_bestTune) %>%
  fit(data=bikeTrain_registered)







## SPLIT DATA FOR CASUAL
bikeTrain_casual <- bikeTrain %>%
  select(-registered, -count) %>%
  mutate(casual = ifelse(casual > 0, log(casual), casual))


## CREATE RECIPE
casual_bike_recipe <- recipe(casual ~ ., data = bikeTrain_casual) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_date(datetime, features="dow") %>%
  step_time(datetime, features="hour") %>% 
  step_date(datetime, features = "month") %>%
  step_date(datetime, features = "year") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## CREATE MODEL (FOREST)
casual_mod <- bart(trees = num_trees) %>%
  set_engine("dbarts") %>%
  set_mode("regression")

## CREATE A WORKFLOW WITH MODEL AND RECIPE
casual_wf <- workflow() %>%
  add_recipe(casual_bike_recipe) %>%
  add_model(casual_mod)

## SET UP GRID OF TUNING VALUES
L <- 5
K <- 5

casual_tuning_grid <- grid_regular(mtry(range(c(1,ncol(bikeTrain_casual)))),
                                   min_n(),
                                   levels = L) ## L^2 total possibilites

## SET UP K-FOLD CV
casual_folds <- vfold_cv(bikeTrain_casual, v = K, repeats = 1)

## FIND BEST TUNING PARAMETERS
casual_CV_results <- casual_wf %>%
  tune_grid(resample = casual_folds,
            grid = casual_tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) ## Or leave metrics NULL

casual_bestTune <- casual_CV_results %>%
  select_best("rmse")

## FINALIZE WORKFLOW AND PREDICT
final_casual_wf <- casual_wf %>%
  finalize_workflow(casual_bestTune) %>%
  fit(data=bikeTrain_casual)

## PREDICTION AND COMBINATION
registered_test_preds <- predict(final_registered_wf, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>%
  select(datetime, .pred) %>%
  rename(registered = .pred) %>%
  mutate(registered=exp(registered)) %>% 
  mutate(registered=pmax(0, registered)) %>% 
  mutate(datetime=as.character(format(datetime))) 


# Predict casual rentals for the test data
casual_test_preds <- predict(final_casual_wf, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>%
  select(datetime, .pred) %>%
  rename(casual = .pred) %>%
  mutate(casual=exp(casual)) %>% 
  mutate(casual=pmax(0, casual)) %>% 
  mutate(datetime=as.character(format(datetime)))

# Sum the predictions
combined_test_preds <- inner_join(registered_test_preds, casual_test_preds, by = "datetime") %>%
  mutate(count = registered + casual) %>%
  select(datetime, count)

## CSV FOR KAGGLE
vroom_write(x=combined_test_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/TestPreds_Split_Bart.csv", delim=",")
