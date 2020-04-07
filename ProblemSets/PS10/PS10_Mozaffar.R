library(rpart)
library(e1071)
library(kknn)
library(nnet)
library(mlr)
library(sets)
set.seed(100)

income <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
names(income) <- c("age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours","native.country","high.earner")
income$native.country <- NULL
income$fnlwgt         <- NULL
income$age            <- as.numeric(income$age)
income$hours          <- as.numeric(income$hours)
income$education.num  <- as.numeric(income$education.num)
income$capital.gain   <- as.numeric(income$capital.gain)
income$capital.loss   <- as.numeric(income$capital.loss)
levels(income$education) <- list(Advanced = c("Masters,","Doctorate,","Prof-school,"), Bachelors = c("Bachelors,"), "Some-college" = c("Some-college,","Assoc-acdm,","Assoc-voc,"), "HS-grad" = c("HS-grad,","12th,"), "HS-drop" = c("11th,","9th,","7th-8th,","1st-4th,","10th,","5th-6th,","Preschool,"))
levels(income$marital.status) <- list(Married = c("Married-civ-spouse,","Married-spouse-absent,","Married-AF-spouse,"), Divorced = c("Divorced,","Separated,"), Widowed = c("Widowed,"), "Never-married" = c("Never-married,"))
levels(income$race) <- list(White = c("White,"), Black = c("Black,"), Asian = c("Asian-Pac-Islander,"), Other = c("Other,","Amer-Indian-Eskimo,"))
levels(income$workclass) <- list(Private = c("Private,"), "Self-emp" = c("Self-emp-not-inc,","Self-emp-inc,"), Gov = c("Federal-gov,","Local-gov,","State-gov,"), Other = c("Without-pay,","Never-worked,","?,"))
levels(income$occupation) <- list("Blue-collar" = c("?,","Craft-repair,","Farming-fishing,","Handlers-cleaners,","Machine-op-inspct,","Transport-moving,"), "White-collar" = c("Adm-clerical,","Exec-managerial,","Prof-specialty,","Sales,","Tech-support,"), Services = c("Armed-Forces,","Other-service,","Priv-house-serv,","Protective-serv,"))
n <- nrow(income)
train <- sample(n, size = .8*n)
test  <- setdiff(1:n, train)
income.train <- income[train,]
income.test  <- income[test, ]
View(income)

#Trees
theTask <- makeClassifTask(id = "taskname", data = income.train, target = "high.earner")
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
treeAlg<-makeLearner("classif.rpart",predict.type="response")
modelParams <- makeParamSet(makeIntegerParam("minsplit",lower=10,upper=50),makeIntegerParam("minbucket",lower=5,upper=50),makeNumericParam("cp",lower=.001,upper=.2))
tuneMethod <- makeTuneControlRandom(maxit = 10)
tunedModel <- tuneParams(learner = treeAlg,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = list(f1,gmean),     
                         par.set = modelParams,
                         control = tuneMethod,
                         show.info = TRUE)
predAlg <- setHyperPars(learner=treeAlg, par.vals = tunedModel$x)
resample(predAlg,theTask,resampleStrat,measures=f1)
finalModel <- train(learner = treeAlg, task = theTask)
prediction <- predict(finalModel, newdata = income.test)
measureF1(prediction$data$truth,prediction$data$response, positive=">50K")
measureGMEAN(prediction$data$truth,prediction$data$response, positive=">50K", negative="<=50K")

#minsplit: 47
#minbucket: 9
#cp: 0.0102
#in sample f1: .8950583
#in sample gmean: 0.6598844
#out of sample F1: .5758662
#out of sample GMean: 0.6650725

#logistic regression
logReg<-makeLearner("classif.glmnet",predict.type="response")
theTask <- makeClassifTask(id = "taskname", data = income.train, target = "high.earner")
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
modelParams <- makeParamSet(makeNumericParam("lambda",lower=0,upper=3),makeNumericParam("alpha",lower=0,upper=1))
tuneMethod <- makeTuneControlRandom(maxit = 10)
tunedModel <- tuneParams(learner = logReg,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = list(f1, gmean),     
                         par.set = modelParams,
                         control = tuneMethod,
                         show.info = TRUE)
predAlg <- setHyperPars(learner=logReg, par.vals = tunedModel$x)
resample(predAlg,theTask,resampleStrat,measures=list(f1,gmean))
finalModel <- train(learner = logReg, task = theTask)
measureF1(prediction$data$truth,prediction$data$response, positive=">50K")
measureGMEAN(prediction$data$truth,prediction$data$response, positive=">50K", negative="<=50K")

#lambda=0.181; 
#alpha=0.0273 : 
#in sample f1=0.8945529
#in sample gmean=0.6120897
#out of sample f1 = .5758662
#out of sample gmean = .6650725

#Neural network
theTask <- makeClassifTask(id = "taskname", data = income.train, target = "high.earner")
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
neuralNet<-makeLearner("classif.nnet",predict.type="response")
modelParams <- makeParamSet(makeIntegerParam("size",lower=1,upper=10),makeNumericParam("decay",lower=.1,upper=.5),makeIntegerParam("maxit",lower=1000,upper=1000))
tuneMethod <- makeTuneControlRandom(maxit = 10)
tunedModel <- tuneParams(learner = neuralNet,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = list(f1,gmean),       
                         par.set = modelParams,
                         control = tuneMethod,
                         show.info = TRUE)
predAlg <- setHyperPars(learner=neuralNet, par.vals = tunedModel$x)
resample(predAlg,theTask,resampleStrat,measures=list(f1,gmean))
finalModel <- train(learner = predAlg, task = theTask)
prediction <- predict(finalModel, newdata = income.test)
measureF1(prediction$data$truth,prediction$data$response, positive=">50K")
measureGMEAN(prediction$data$truth,prediction$data$response, positive=">50K", negative="<=50K")


#size=8; 
#decay=0.362; 
#maxit=1000 
# in sample f1=0.9050971,
# in sample gmean=0.7526343
# out of sample f1: .606198
#out of sample g mean: .6963027

#Naive Bayes
theTask <- makeClassifTask(id = "taskname", data = income.train, target = "high.earner")
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
naiveBayes<-makeLearner("classif.naiveBayes",predict.type="response")
tuneMethod <- makeTuneControlRandom(maxit = 10)
tunedModel <- tuneParams(learner = predAlg,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = rmse,       
                         par.set = modelParams,
                         control = tuneMethod,
                         show.info = TRUE)
predAlg <- setHyperPars(learner=predAlg, par.vals = tunedModel$x)
resample(naiveBayes,theTask,resampleStrat,measures=list(f1,gmean))
finalModel <- train(learner = naiveBayes, task = theTask)
prediction <- predict(finalModel, newdata = income.test)
measureF1(prediction$data$truth,prediction$data$response, positive=">50K")
measureGMEAN(prediction$data$truth,prediction$data$response, positive=">50K", negative="<=50K")

#out of sample f1: .6245825
#out of sample gmeans: .7343107


#k nearest neighbors
theTask <- makeClassifTask(id = "taskname", data = income.train, target = "high.earner")
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
kNearNeighbors<-makeLearner("classif.kknn",predict.type="response")
modelParams <- makeParamSet(makeIntegerParam("k",lower=1,upper=30))
tuneMethod <- makeTuneControlRandom(maxit = 10)
tunedModel <- tuneParams(learner = kNearNeighbors,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = list(f1,gmean),      
                         par.set = modelParams,
                         control = tuneMethod,
                         show.info = TRUE)
predAlg <- setHyperPars(learner=kNearNeighbors, par.vals = tunedModel$x)
resample(predAlg,theTask,resampleStrat,measures=list(f1, gmean))
finalModel <- train(learner = predAlg, task = theTask)
prediction <- predict(finalModel, newdata = income.test)
measureF1(prediction$data$truth,prediction$data$response, positive=">50K")
measureGMEAN(prediction$data$truth,prediction$data$response, positive=">50K", negative="<=50K")


#k=30 
# in sample f1=0.8972547 
# in sample gmean=0.7489182
# out of sample f1 = 0.6480252
#out of sample gmean = .7413514

#Support Vector Machine
theTask <- makeClassifTask(id = "taskname", data = income.train, target = "high.earner")
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
SVN<-makeLearner("classif.svm",predict.type="response")
modelParams <- makeParamSet(makeDiscreteParam("kernel",value="radial"),makeDiscreteParam("cost", values=c(2^-2,2^-1,2^0,2^1,2^2,2^10)),makeDiscreteParam("gamma", values=c(2^-2,2^-1,2^0,2^1,2^2,2^10)))
tuneMethod <- makeTuneControlRandom(maxit = 10)
tunedModel <- tuneParams(learner = SVN,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = list(f1,gmean),       
                         par.set = modelParams,
                         control = tuneMethod,
                         show.info = TRUE)
predAlg <- setHyperPars(learner=predAlg, par.vals = tunedModel$x)
resample(predAlg,theTask,resampleStrat,measures=list(f1,gmean))
finalModel <- train(learner = predAlg, task = theTask)
prediction <- predict(finalModel, newdata = income.test)
measureF1(prediction$data$truth,prediction$data$response, positive=">50K")
measureGMEAN(prediction$data$truth,prediction$data$response, positive=">50K", negative="<=50K")


