##### This is the code to fit prediction models using three types of document matrix
setwd("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec")
load("HouseHearings_submit.Rdata")  # load hearing data
load("doc2_sum62.Rdata")  # load the best doc2vec matrix chosen from "Rcode2_GridSearch_KSVM.R"
load("label_sample.Rdata")
load("label_test.Rdata")
load("sample_id.Rdata")
load("test_id.Rdata")
memory.limit(size=50000)


library(SuperLearner)
library(quanteda)



########################################################################
### Compute weights
# 1) First 1000
# the number of hearings in the 114th: 1448
only114 <- hearing_noagg[which(hearing_noagg$congress=="114"),]
uni <- unique(only114$file_name)
length(uni)  


# total number of hearings: 12820
uni2 <- unique(hearing_noagg$file_name) 
length(uni2) 


# average number of speeches per hearing: 80.08401
avgsp=nrow(hearing_noagg)/length(uni2)


# probability to be included in the first 1000: 0.008623541
w1000 = (50/length(uni))*(20/avgsp) #approx. (50/1448)*(20/80.08401): I chose 50 hearings from the 114th and then 20 speeches were selected from each hearing on average. 


# 3) Third 2000
# I chose 2000 hearings from the 12820-50=12770, and then chose one speech from each hearing
w2000 = (2000/(length(uni2)-50))*(1/avgsp)  # =(2000/12770)*(1/80.08401) = 0.00195566


# Compute the weights
# Given that the first 1000 receives weight of 1, the third 2000 receives: 4.40953
w1000/w2000



####### Define weights (1:4.4) ######### (See below to see how weights are calculated)
first1000 <- which(data$dataset_new==1)
third2000 <- which(data$dataset_new==3)

weight <- rep(1, length(sample_id))
weight[sample_id %in% third2000] <- 4.4
#save(weight, file="weight.Rdata")


############################## 1) Using doc2vec matrix ################################################  
m_sample <- as.data.frame(doc2_sum[sample_id,])  
m_test <- as.data.frame(doc2_sum[test_id,])   


model_list <- c("svm", "svm", "ksvm", "ksvm", "glmnet", "glmnet", "randomForest", "randomForest", "bayesglm", "gbm", "gbm", "lm", "dbarts")
option_list_weight <- c("", ",cost=10", "", ", epsilon=0.5", ", obsWeights = weight_train", ", obsWeights = weight_train ,nlambda = 200", "", ",nodesize = 20", ", obsWeights = weight_train", ", obsWeights = weight_train",
                        ", obsWeights = weight_train, shrinkage=.01", ", obsWeights = weight_train", ", obsWeights = weight_train") # I added weight to glmnet, bayesglm, gbm, lm and dbarts. svm, ksvm and randomForest don't seem to have a weighting option.
option_list_weight_test <- c("", ",cost=10", "", ", epsilon=0.5", ", obsWeights = weight", ", obsWeights = weight ,nlambda = 200", "", ",nodesize = 20", ", obsWeights = weight", ", obsWeights = weight",
                             ", obsWeights = weight, shrinkage=.01", ", obsWeights = weight", ", obsWeights = weight") # I added weight to glmnet, bayesglm, gbm, lm and dbarts. svm, ksvm and randomForest don't seem to have a weighting option.


rmse_sam <- rep(0,length(model_list))
cor_sam <- rep(0,length(model_list))
rmse_test <- rep(0,length(model_list))   
cor_test <- rep(0,length(model_list))    
results <- data.frame(model_list, option_list, rmse_sam, cor_sam, rmse_test, cor_test)   # adjust


for(j in 1:length(model_list)){
  
  rmse <- rep(0,10)
  cor <- rep(0,10)
  pred_sample <- c(1:nrow(m_sample))
  p <- rep(1:10, each=nrow(m_sample)/10) # Define positions for the train+validation set to be divided into 10 pieces for 10-fold CV
  
  
  for(i in 1:10){
    train <- m_sample[which(p!=i),]
    label_train <- label_sample[which(p!=i)]
    weight_train <- weight[which(p!=i)]
    
    valid <- m_sample[which(p==i),]
    label_valid <- label_sample[which(p==i)]
    
    set.seed(150)  # set seed to reproduce the same model fit 
    #  model <- eval(parse(text=paste("SL.",model_list[j],"(label_train, train, valid, family = gaussian()",option_list[j],")",sep="")))
    model <- eval(parse(text=paste("SL.",model_list[j],"(label_train, train, valid, family = gaussian()",option_list_weight[j],")",sep="")))
    rmse[i] <- sqrt(mean((label_valid - model$pred)^2)) 
    cor[i] <- cor(label_valid, model$pred)
    
    pred_sample[which(p==i)] <- model$pred
  }
  results$rmse_sam[j] <- sum(rmse)/10   # For a j-th model
  results$cor_sam[j] <- sum(cor)/10   # For a j-th model
  
  
  pred_sample_file <- paste("pred_sample",j,".Rdata",sep="")
  save(pred_sample, file=pred_sample_file)
  
  
  ### Predict the 300 virgin paragraphs using the same model    
  model <- eval(parse(text=paste("SL.",model_list[j],"(label_sample, m_sample, m_test, family = gaussian()",option_list_weight_test[j],")",sep="")))
  results$rmse_test[j] <- sqrt(mean((label_test - model$pred)^2)) 
  results$cor_test[j] <- cor(label_test, model$pred)
  
  
  pred_test <- model$pred
  pred_test_file <- paste("pred_test",j,".Rdata",sep="")
  save(pred_test, file=pred_test_file)
  
  
  print(paste("done for processing for:", pred_test_file))
}

#save(results, file="results_doc2vec_weight.Rdata")




