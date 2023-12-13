library(vroom)
library(dplyr)
library(ggplot2)
library(corrplot)
library(tidymodels)
library(ranger)

# Load Data
train <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/training.csv')
raw_test <- vroom('C:/Users/jbhil/Fall 2023/STAT_346/dont_get_kicked/test.csv')

# Identify categorical columns
categorical_cols <- train %>%
  select_if(is.character) %>%
  names()

# Identify numeric columns
numerical_cols <- train %>%
  select_if(is.numeric) %>%
  names()

# Print the results
cat("Categorical columns:", toString(categorical_cols), "\n")
cat("Numeric columns:", toString(numerical_cols), "\n")

# Quick glance at data
head(train)
str(train)


# Create a data frame with categorical variable counts
category_counts_df <- data.frame(
  Variable = character(),
  Count = integer(),
  stringsAsFactors = FALSE
)

for (col in categorical_cols) {
  category_counts_df <- rbind(category_counts_df, data.frame(Variable = col, Count = length(unique(train[[col]]))))
}

# Sort the data frame by Count in descending order
category_counts_df <- category_counts_df %>%
  arrange(desc(Count))

# Plot the bar chart
ggplot(category_counts_df, aes(x = Count, y = reorder(Variable, Count))) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Categorical variable counts") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10))
