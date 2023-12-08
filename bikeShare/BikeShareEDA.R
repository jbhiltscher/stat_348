library(readr)
library(ggplot2)
library(patchwork)
library(wesanderson)
train <- read_csv("Fall 2023/STAT_346/train.csv")

dplyr::glimpse(train)

DataExplorer::plot_correlation(train)

plot1 <- ggplot(data = train, aes(x=temp, y=casual, color = as.factor(season))) +
  geom_point()

plot2 <- ggplot(data = train, aes(x = humidity, y = registered, color = as.factor(season))) + 
  geom_point()

plot3 <- ggplot(data = train, aes(x=weather, fill=as.factor(season))) +
  geom_bar(color="black") +
  scale_fill_brewer() +
  guides(fill = guide_legend(title = "Season"))

plot4 <- DataExplorer::plot_correlation(train)

(plot1 + plot2) / (plot3 + plot4)
