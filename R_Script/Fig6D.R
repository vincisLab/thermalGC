# Linear regression Figure6D - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.
# install the library readxl if needed.

library(readxl)
# change the filepath
cold <- read_excel("/Users/robertovincis/Documents/GitHub/thermalGC/Data/data_Figure6D_cold.xls")
lmCold = lm(CS~CW, data = cold) #Create the linear regression
summary(lmCold) #Review the results

hot <- read_excel("/Users/robertovincis/Documents/GitHub/thermalGC/Data/data_Figure6D_hot.xls")
lmHot = lm(HS~HW, data = hot) #Create the linear regression
summary(lmHot) #Review the results