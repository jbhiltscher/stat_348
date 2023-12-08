library(vroom)
library(tidymodels)
library(dplyr)
library(modeltime)
library(timetk)
library(progress)

train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/train.csv")
test <- vroom ("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/test.csv")

## RECIPE
my_recipe <- recipe(sales ~., data = train) %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))

#es_model <- exp_smoothing() %>%
#  set_engine("ets") %>%
#  fit(sales~date, data= training(cv_split))

#cv_results <- modeltime_calibrate(es_model,
#                                 new_data = testing(cv_split))

#cv_results %>%
#  modeltime_forecast(
#    new_data = testing(cv_split),
#    actual_data = cross_val_data
#  ) %>%
#  plot_modeltime_forecast(.interactive=TRUE)
  
#cv_results %>%
#  modeltime_accuracy() %>%
#  table_modeltime_accuracy(.interactive = FALSE)

#es_fullfit <- cv_results %>%
# modeltime_refit(data=storeItemTrain)

#es_preds <- es_fullfit %>%
#  modeltime_forecast(h="3 months") %>%
#  rename(date=.index, sales=.value) %>%
#  select(date, sales) %>%
#  full_join(., y=storeItemTest, by="date") %>%
#  select(id, sales)

#es_fullfit %>% modeltime_forecast(h = "3 months", actual_data = train) %>%
#  plot_modeltime_forecast(.interactive=FALSE)


nStores <- max(train$store)
nItems <- max(train$item)

n_total <- nStores * nItems
pb <- progress_bar$new(
  format = "  progress [:bar] :percent elapsed=:elapsed, eta=:eta",
  total = n_total
)

for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)
    
    ## Fit storeItem models here
    cv_split <- time_series_split(storeItemTrain, assess = "3 months", cumulative = TRUE)
    
    es_model <- exp_smoothing() %>%
      set_engine("ets") %>%
      fit(sales~date, data= training(cv_split))
    
    cv_results <- modeltime_calibrate(es_model,
                                      new_data = testing(cv_split))
    es_fullfit <- cv_results %>%
      modeltime_refit(data=storeItemTrain)
    
    
    ## Predict storeItem sales
    es_preds <- es_fullfit %>%
      modeltime_forecast(h="3 months") %>%
      rename(date=.index, sales=.value) %>%
      select(date, sales) %>%
      full_join(., y=storeItemTest, by="date") %>%
      select(id, sales)
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- es_preds
    } else {
      all_preds <- bind_rows(all_preds, es_preds)
    }
    
    # Increment the progress bar
    pb$tick()
    
  }
}
# Close the progress bar
pb$close()