library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)

item <- vroom::vroom("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/train.csv")
itemTest <- vroom::vroom("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/test.csv")

nStores <- max(item$store)
nItems <- max(item$item)

## Define the workflow

# Recipe
my_recipe <- recipe(sales~., data=item) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())

# Model
boosted_model <- boost_tree(tree_depth=2, #Determined by random store-item combos
                            trees=1000,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

# Workflow
boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model)

## Double Loop over all store-item combos
for(s in 1:nStores){
  for(i in 1:nItems){
    
    ## Subset the data
    train <- item %>%
      filter(store==s, item==i)
    test <- itemTest %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast
    fitted_wf <- boost_wf %>%
      fit(data=train)
    preds <- predict(fitted_wf, new_data=test) %>%
      bind_cols(test) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save the results
    if(s==1 && i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds,
                             preds)
    }
    
  }
}

vroom_write(all_preds, file = "C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/submission.csv", delim = ',')
