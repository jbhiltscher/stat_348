library(vroom)
library(tidymodels)
library(dplyr)
library(modeltime)
library(timetk)
library(prophet)

train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/train.csv")
test <- vroom ("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/test.csv")

## RECIPE
my_recipe <- recipe(sales ~., data = train) %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))

nStores <- max(train$store)
nItems <- max(train$item)

for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)
    
    ## Fit storeItem models here
    cv_split <- time_series_split(storeItemTrain, assess = "3 months", cumulative = TRUE)
    
    prophet_model <- prophet_reg() %>%
      set_engine("prophet")
    
    prophet_wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(prophet_model) %>%
      fit(data=training(cv_split))
    
    cv_results <- modeltime_calibrate(prophet_wf,
                                      new_data = testing(cv_split))
    fullfit <- cv_results %>%
      modeltime_refit(data=storeItemTrain)
    
    ## Predict storeItem sales
    preds <- fullfit %>%
      modeltime_forecast(new_data = storeItemTest,
                         actual_data = storeItemTrain) %>%
      rename(date=.index, sales=.value) %>%
      filter(.key =='prediction') %>% 
      mutate(sales = round(sales, digits = 0)) %>%
      select(date, sales) %>%
      full_join(., y=storeItemTest, by='date') %>%
      select(id, sales)
    
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
  }
}

vroom_write(all_preds, file = "prophet_preds.csv", delim = ',')