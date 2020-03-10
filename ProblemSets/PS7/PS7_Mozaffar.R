#problem 3
library(tidyverse)
library(mice)
install.packages("stargazer")
library(stargazer)

#problem 4
wages<-read.csv("https://raw.githubusercontent.com/yaseenmozaffar/DScourseS20/master/ModelingOptimization/wages.csv")

#problem 5
wages<-drop_na(wages, hgc)
wages<-drop_na(wages,tenure)

#problem 6
stargazer(wages)
md.pattern(wages)

#problem 7a
wages.listwise<- drop_na(wages,logwage)
listwise.regress<- lm(logwage~ hgc + college + tenure + (tenure^2) + age + married, data=wages)

#problem 7b
logwage.mean<- mean(wages$logwage)
wages.mean.impt<-wages
wages.mean.impt$logwage[is.na(wages.mean.impt$logwage)]<-logwage.mean
mean.impt.regress<- lm(logwage~ hgc + college + tenure + (tenure^2) + age + married, data=wages.mean.impt)

#problem 7c
wages.single.impt<-wages
wages.single.impt$logwage[is.na(wages.single.impt$logwage)]<-listwise.regress$fitted.values
single.impt.regress<- lm(logwage~ hgc + college + tenure + (tenure^2) + age + married, data=wages.single.impt)

#problem 7d
wages.mice<-wages
mice.impt<-mice(wages.mice, seed = 12345)
summary(mice.impt)
fit = with(mice.impt, lm(logwage~ hgc + college + tenure + (tenure^2) + age + married))
round(summary(pool(fit)),2)

#problem 7e
stargazer(listwise.regress,mean.impt.regress,single.impt.regress)

