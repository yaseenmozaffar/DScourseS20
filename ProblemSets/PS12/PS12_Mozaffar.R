library(sampleSelection)
library(tidyverse)
library(stargazer)
library(mice)

#4
wages<-as.data.frame(read.csv("https://raw.githubusercontent.com/yaseenmozaffar/DScourseS20/master/Structural/wages12.csv"))
view(wages)

#5
wages$college<-as.factor(wages$college)
wages$married<-as.factor(wages$married)
wages$union<-as.factor(wages$union)

#6
stargazer(wages)
md.pattern(wages)

#7a
wages.listwise<- drop_na(wages,logwage)
listwise.regress<- lm(logwage~ hgc + union + college + exper + (exper^2), data=wages)

#7b
logwage.mean<- mean(wages$logwage)
wages.mean.impt<-wages
wages.mean.impt$logwage[is.na(wages.mean.impt$logwage)]<-logwage.mean
mean.impt.regress<-lm(logwage~ hgc + union + college + exper + (exper^2), data=wages)
#7c
wages$valid<-as.numeric(is.na(wages$logwage)==FALSE)
wages$logwage[is.na(wages$logwage)]<-0
wages.heckit<-selection(selection = valid ~ hgc + union + college + exper + married + kids ,
                    outcome = logwage ~ hgc + union + college + exper + I ( exper ^ 2 ) ,
                    data = wages, method = " 2step " )
stargazer(listwise.regress,mean.impt.regress,wages.heckit)

#8
union.probit<-glm(union ~ hgc + college + exper + married + kids, family=binomial(link="probit"), data=wages)
summary(union.probit)
wages$predProbit <- predict(union.probit, newdata = wages, type = "response")
summary(wages$predProbit)
#9
union.probit$coefficients["kids"]<-0
union.probit$coefficients["married"]<-0
wages$counterfact<-predict(union.probit, newdata = wages, type = "response")

summary(wages$counterfact)
counterfact$coefficients["married"]
stargazer(wages)
