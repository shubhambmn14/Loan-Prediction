RANDOM FOREST IMPLEMENTATION

rm(list = ls())
library(ggplot2)
library(randomForest)
library(caret)#For creating confusion matrix
library(VIM) #Missing 
setwd("D://Sunstone//kaggle//Loan Prediction -III")
train <- read.csv("train.csv", header = TRUE)

test <- read.csv("test.csv", header = T)
head(train)
str(train)
dim(train) # 614 13
summary(train)
#There are lot of missing values in the whole train data.
#before start treating the missing values lets combine the test data and train data together
test$Loan_Status <- NA
combi <- rbind(train,test)
summary(combi)

#It makes more sense to add the applicant and coapplicant Income and use it as one variable
combi$ApplicantTotalIncome <- combi$ApplicantIncome + combi$CoapplicantIncome
combi <- combi[,-c(7,8)]#Removing the ApplicantIncome & CoapplicantIncome

#Studying some pattern in the data - Q.1 Does bank prefer any particular Gender?
prop.table(table(train$Gender, train$Loan_Status),1)

#             N         Y
#       0.3846154 0.6153846
#Female 0.3303571 0.6696429
#Male   0.3067485 0.6932515
#We can see there is not much gender bias in the loan approval status.

#--------Q.2 Does being married help in getting loan?---------#
prop.table(table(train$Married , train$Loan_Status),2)
#         N         Y
#    0.0000000 1.0000000
#No  0.3708920 0.6291080
#Yes 0.2839196 0.7160804

#We can see that if someone is married their is 71% chance that he will get the loan
#While a non-married person has it much lower chance of 63%

#-------- Q.2 Does being graduate helps in getting the loan?---------#

prop.table(table(train$Education, train$Loan_Status),1)

#                  N         Y
#Graduate     0.2916667 0.7083333
#Not Graduate 0.3880597 0.6119403
#There is definitely a difference if some is a graduate or not. This makes sense aswell

#-------Q.3 Does being self employed lower ur chance to secure the loan?-------#
prop.table(table(train$Self_Employed, train$Loan_Status),1)
#There is no difference here.

#Q.4------How does property Area influence loan? Does it goes more for Urban?-----#
prop.table(table(train$Property_Area, train$Loan_Status),1)*100

#              N        Y
#Rural     38.54749 61.45251
#Semiurban 23.17597 76.82403
#Urban     34.15842 65.84158

#Looks like banks are approving more loans in the semiurban areas. Rural sees most rejection

#-------Q.5: Credit History has good correlation with loan approval?--------#

prop.table(table(train$Credit_History, train$Loan_Status),1)*100
#      N         Y
#0 92.134831  7.865169
#1 20.421053 79.578947
#WOW!! Now thats an evidence. People whose credit history doesnt meet guidelines are getting
#rejections 92% of the times. It would be interesting to see who are the other 8% still
#getting the loans? Else if your credit history is good your approval chances are 80%

prop.table(table(train$Dependents, train$Loan_Status),1)*100

#        N        Y
#   40.00000 60.00000
#0  31.01449 68.98551
#1  35.29412 64.70588
#2  24.75248 75.24752
#3+ 35.29412 64.70588


Lets draw one scatter plot to see how Applicant Income and Loan Amount are related

plot1 <- (ggplot(train, aes(y = ApplicantIncome, x= LoanAmount, col = Loan_Status)) 
                + geom_point(position = "jitter", size=3, alpha = 0.6) 
                + geom_smooth(se = FALSE, method = "lm") 
                + facet_grid(~Loan_Status)
                + labs(x = "Loan Amount ('000)", y = "Applicant income", title = "Scatterplot - Income Vs Loan Amount seeked")
                + theme_bw()) 
      
plot2 <- (ggplot(train, aes(x = ApplicantIncome+CoapplicantIncome, fill = Loan_Status))
          + geom_histogram(col = "black",fill = "green", bins = 200,breaks=seq(0, 80000, by = 1000))
          + labs(x = "Total Income", title = "Distribution of Total Income")
          + facet_grid(Loan_Status~.)
          + theme_bw())
#Rplot3

plot3 <- (ggplot(train, aes(y = ApplicantIncome + CoapplicantIncome, x = Loan_Status))
          + geom_boxplot()
          + labs(x = "loanStatus", title = "Distribution of Total Income")
          + facet_grid(.~Self_Employed)
          + coord_flip())

plot4 <- (ggplot(train, aes(y = LoanAmount, x = Loan_Status)) #Rplot4
          + geom_point(size = 3, alpha = 0.6)
          + facet_grid(Credit_History ~ .)
          + coord_flip())

#Again Checking on the summary for combi
summary(combi)
combi$Credit_History <- as.factor(combi$Credit_History)
train$Loan_Status <- as.character(train$Loan_Status)
train$Loan_Status <- ifelse(train$Loan_Status == 'Y', 1, 0)
aggregate(Loan_Status ~ Gender + Property_Area, data = train, FUN = function(x){sum(x)/length(x)})
#Looks like Credit history is the most important variable but we have 79 not available
#I believe that someone with more Income would have more chance of meeting credit history req.
plot5 <- (ggplot(combi, aes(x = Credit_History, y = ApplicantTotalIncome))
          + geom_boxplot()
          + coord_flip())

plot6 <- (ggplot(combi, aes(x = Loan_Amount_Term, y = LoanAmount, col = Loan_Status))
          + geom_point(size = 3, aes(shape = Loan_Status), alpha = 0.7, position = "jitter")
          + geom_smooth(se = F, method = "lm"))

#Not much of a pattern here.


#Removing the loan ID coloumn
combi$Loan_ID <- NULL

summary(combi)
combi <- combi[,c(1:9,11,10)]
#Imputing the values using kNN
a <- kNN(combi, variable = c("LoanAmount","Loan_Amount_Term","Credit_History"), k =5)
sum(is.na(combi_new))
combi_new <- a[,-c(12,13,14)]
str(combi_new)

combi_new$Loan_Amount_Term <- as.factor(combi_new$Loan_Amount_Term)
combi_new$Gender <- as.character(combi_new$Gender)
combi_new$Gender[combi_new$Gender == ""] <- NA
combi_new$Gender <- as.factor(combi_new$Gender)

combi_new$Married <- as.character(combi_new$Married)
combi_new$Married[combi_new$Married == ""] <- NA
combi_new$Married <- as.factor(combi_new$Married)

combi_new$Self_Employed <- as.character(combi_new$Self_Employed)
combi_new$Self_Employed[combi_new$Self_Employed == ""] <- NA
combi_new$Self_Employed <- as.factor(combi_new$Self_Employed)

combi_new$Dependents <- as.character(combi_new$Dependents)
combi_new$Dependents[combi_new$Dependents == ""] <- NA
combi_new$Dependents <- as.factor(combi_new$Dependents)

combi_new <- kNN(combi_new, variable = c("Gender","Married","Dependents","Self_Employed"),k=5)
combi_new <- combi_new[,1:11]#Now we have no NAs

train_updated <- combi_new[1:614,]
test_updated <- combi_new[615:981,]


fit <- randomForest(Loan_Status ~ ., data= train_updated, ntree = 500, importance = T, na.action = na.omit)
varImpPlot(fit)   
plot(fit)
train_updated$PredictedLoan_Status <- predict(fit, train_updated[,1:10])
confusionMatrix(train_updated$PredictedLoan_Status, train_updated$Loan_Status)

#Reference
#Prediction   N   Y
#N 167  25
#Y   0 422

#Accuracy : 0.9593 

test$Loan_Status <- predict(fit, test_updated)
submission <- test[,c(1,13)]
#First Submission
write.csv(submission, file = "RF1.csv", row.names = T)
print(importance(fit,type = 2)) 
print(fit)
#--------Submission 2----------#
train_updated <- train_updated[,-12]

fit <- randomForest(Loan_Status ~ ., data= train_updated, 
                    ntree = 500, 
                    importance = T,
                     mtry = 2)

fit
varImpPlot(fit)     
test$Loan_Status <- predict(fit, test_updated)
submission <- test[,c(1,13)]
#Second Submission
write.csv(submission, file = "RF2.csv", row.names = T)
#0.791677 - Score improved


#-------------Iteration 3-----------------#
#Removing the Self_Employed variable since it has the min decrease in the Gini score

train_updated <- train_updated[,-5]

fit <- randomForest(Loan_Status ~ ., data= train_updated, 
                    ntree = 500, 
                    importance = T,
                    mtry = 2)

fit
varImpPlot(fit)     
test$Loan_Status <- predict(fit, test_updated)
submission <- test[,c(1,13)]
#Third Submission
write.csv(submission, file = "RF3.csv", row.names = T)

#--------Iteration 4----------#
#Removing the Gender variable as well_numbered
train_updated <- train_updated[,-1]

fit <- randomForest(Loan_Status ~ ., data= train_updated, 
                    ntree = 500, 
                    importance = T,
                    mtry = 2)

fit
varImpPlot(fit)     
test$Loan_Status <- predict(fit, test_updated)
submission <- test[,c(1,13)]
#Third Submission
write.csv(submission, file = "RF4.csv", row.names = T)
summary(combi)
boxplot(combi$ApplicantTotalIncome)
