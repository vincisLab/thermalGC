# One Way Anova Figure5E - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.
# install the library dplyr and ggpubr if needed.

library(dplyr)
# change the path if needed
PATH <- "/Users/robertovincis/Documents/GitHub/thermalGC/Data/data_Figure5E.csv"
df <- read.csv(PATH) %>%
  select(-X) %>% 
  mutate(Position = factor(Position))
glimpse(df)

library("ggpubr")
ggline(df, x = "Position", y = "Dec",
      add = c("mean_se", "jitter"),
      order = c("zero", "one", "two","three"),
      ylab = "Decoding Accuracy", xlab = "Position")


# Compute the analysis of variance
res.aov <- aov(Dec ~ Position, data = df)
# Summary of the analysis
summary(res.aov)