# T test Figure 8D - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.
# install the library readxl if needed.

library(readxl)

SI <- read_excel("/Users/robertovincis/Documents/GitHub/thermalGC/Data/data_Figure8D.xls")

# test for equality of variance
eq_var = var.test(SI$TT, SI$OT, alternative = "two.sided")


stat = t.test(SI$TT, SI$OT, alternative = "l", var.equal = TRUE)
