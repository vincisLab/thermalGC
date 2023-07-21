# One Way Anova Figure2D - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.

library(dplyr)
# adjust the path in the next line
PATH <- "~Data/data_Figure2D.csv"
df <- read.csv(PATH) %>%
  select(-X) %>% 
  mutate(Stim = factor(Stim))
glimpse(df)

kruskal.test(Lat ~ Stim, data = df)
