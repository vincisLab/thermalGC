# prepare for the chi square test and Marascuilo's multicomparison
# total number of themperature-selective neurons = 179

# 1 ------------------------ Chi-squared test -------------------------------
values <-c(16,25,138)
res <- chisq.test(values, p = c(1/3, 1/3, 1/3))

# 1 ------------------Marascuilo multicomparison -------------------------------
## Set the proportions of interest.
## Values should be > than critical range!
p = c(0.089, 0.139, 0.770)
N = length(p)
value = critical.range = c()

## Compute critical values.
for (i in 1:(N-1))
{ for (j in (i+1):N)
{
  value = c(value,(abs(p[i]-p[j])))
  critical.range = c(critical.range,
                     sqrt(qchisq(.99,2))*sqrt(p[i]*(1-p[i])/179 + p[j]*(1-p[j])/179))
}
}

round(cbind(value,critical.range),4)
