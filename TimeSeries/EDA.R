library(timetk)
library(ggplot2)
library(vroom)
library(forecast)
library(dplyr)
library(gridExtra)

## LOAD IN THE DATA
train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/train.csv")
test <- vroom ("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/test.csv")

# Create a function to generate the plot for a specific store and item
plot_store_item <- function(store, item) {
  train %>%
    filter(store == store & item == item) %>%
    pull(sales) %>%
    ggAcf(.) +
    ggtitle(paste("Store", store, "-", "Item", item))
}

# Create a 2x2 panel using gridExtra
combined_plot <- grid.arrange(
  plot_store_item(1, 2),
  plot_store_item(3, 5),
  plot_store_item(5, 1),
  plot_store_item(2, 7),
  ncol = 2
)
ggsave("C:/Users/jbhil/Fall 2023/STAT_346/TimeSeries/combined_plot.png", combined_plot, width = 10, height = 8, units = "in")
