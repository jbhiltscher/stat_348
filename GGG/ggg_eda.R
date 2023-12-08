library(vroom)
library(dplyr)
library(ggplot2)
library(ggmosaic)

ggg_test <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/test.csv")

ggg_train <- vroom("C:/Users/jbhil/Fall 2023/STAT_346/GGG/train.csv")

ggplot(data = ggg_train, aes(x=type, y=bone_length)) +
  geom_boxplot()

ggplot(data = ggg_train) +
  geom_mosaic(aes(x=product(color), fill=type))
