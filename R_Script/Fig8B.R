# T test Figure 8B (left) - Bouaichi, Odegaard, Neese and Vincis
# change the path to the .xls files if needed.
# install the library readxl if needed.

library(readxl)

SI <- read_excel("/Users/robertovincis/VincisLab_Python/Bouaichi_and_Vincis2022/Plos Biology/data_Figure8B.xls")

# test for equality of variance
eq_var = var.test(SI$TT, SI$OT, alternative = "two.sided")


stat = t.test(SI$TT, SI$OT, alternative = "l", var.equal = TRUE)


# proportion test Figure 8B(middle) - Bouaichi, Odegaard, Neese and Vincis
# 1 SI>0 and SI<0.35 (broadly tuned neurons)
a = prop.test(x = c(35, 13), n = c(68, 45), alternative = "g")


# proportion test Figure 8B(right) - Bouaichi, Odegaard, Neese and Vincis
# 1 SI>0.65 and SI<1 (narrowly tuned neurons)
a = prop.test(x = c(5, 9), n = c(68, 45), alternative = "l")

