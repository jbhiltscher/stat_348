library(ranger)
library(vroom)
library(tidymodels)

## LOAD IN DATA AND CLEAN
bikeTrain <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/train.csv")
bikeTest <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/test.csv")

bikeTrain <- bikeTrain %>%
  select(-casual, - registered)
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
  step_date(datetime, features = "year") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

## CREATE MODEL (REGRESSION TREE)
my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                        trees = 20) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## CREATE A WORKFLOW WITH MODEL AND RECIPE
rand_forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_mod)

## SET UP GRID OF TUNING VALUES
L <- 5
K <- 5

tuning_grid <- grid_regular(mtry(range(c(1,ncol(bikeTrain)))),
                            min_n(),
                            levels = L) ## L^2 total possibilites
## SET UP K-FOLD CV
folds <- vfold_cv(bikeTrain, v = K, repeats = 1)

## FIND BEST TUNING PARAMETERS
CV_results <- rand_forest_wf %>%
  tune_grid(resample = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) ## Or leave metrics NULL

bestTune <- CV_results %>%
  select_best("rmse")

## FINALIZE WORKFLOW AND PREDICT
final_wf <- rand_forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bikeTrain)

test_preds <- predict(final_wf, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>%
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>%
  mutate(count=exp(count)) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) # needed for right format to Kaggle

## CSV FOR KAGGLE
vroom_write(x=test_preds, file="C:/Users/jbhil/Fall 2023/STAT_346/TestPreds_Rand_Forest.csv", delim=",")

