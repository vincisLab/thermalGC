# Linear regression Figure5C - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.
# install the library readxl if needed.

library(readxl)

# change the path to the files!!
dec_pos <- read_excel("/Users/robertovincis/Documents/GitHub/thermalGC/Data/data_Figure5C.xls")
lm_dec_pos = lm(Dec~Pos, data = dec_pos) #Create the linear regression
summary(lm_dec_pos) #Review the results