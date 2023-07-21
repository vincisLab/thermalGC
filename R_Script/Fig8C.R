# Wilcoxon test Figure 8C - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.
# install the library readxl if needed.

library(readxl)

dec <- read_excel("/Users/robertovincis/Documents/GitHub/thermalGC/Data/data_Figure8C.xls")

# test for equality of variance
eq_var = var.test(dec$TT, dec$OT, alternative = "two.sided")


stat = t.test(dec$TT, dec$OT, alternative = "g", paired = FALSE, var.equal = TRUE)

# proportion test Figure 8D (middle) - Bouaichi, Odegaard, Neese and Vincis
# 1 SI>0.65 and SI<1 (narrowly tuned neurons)
a = prop.test(x = c(32, 13), n = c(68, 45), alternative = "g")