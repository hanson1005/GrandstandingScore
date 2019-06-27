##### This is the code to fit prediction models using three types of document matrix
setwd("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec")
load("HouseHearings_submit.Rdata")  
load("label_sample.Rdata")
load("label_test.Rdata")
load("sample_id.Rdata")
load("test_id.Rdata")
load("stopwordslist.Rdata")
load("weight.Rdata")
memory.limit(size=50000)


library(SuperLearner)
library(quanteda)
library(text2vec)


################################### 3) Using tfidf doc matrix ############################
load("doc.features_2gram_2000.Rdata")

tfidf = TfIdf$new()
doc.features_tfidf = fit_transform(doc.features, tfidf)
m <- as.matrix(doc.features_tfidf)
m <- as.data.frame(m)
colnames(m) <- make.names(colnames(m), unique = TRUE) # run this before fitting gbm to avoid an error message


m_sample <- m[sample_id,]  
m_test <- m[test_id,]    


model_list <- c("svm", "svm", "ksvm", "ksvm", "glmnet", "glmnet", "randomForest", "randomForest", "bayesglm", "gbm", "gbm", "lm", "dbarts")
option_list_weight <- c("", ",cost=10", "", ", epsilon=0.5", ", obsWeights = weight_train", ", obsWeights = weight_train ,nlambda = 200", "", ",nodesize = 20", ", obsWeights = weight_train", ", obsWeights = weight_train",
                        ", obsWeights = weight_train, shrinkage=.01", ", obsWeights = weight_train", ", obsWeights = weight_train") # I added weight to glmnet, bayesglm, gbm, lm and dbarts. svm, ksvm and randomForest don't seem to have a weighting option.
option_list_weight_test <- c("", ",cost=10", "", ", epsilon=0.5", ", obsWeights = weight", ", obsWeights = weight ,nlambda = 200", "", ",nodesize = 20", ", obsWeights = weight", ", obsWeights = weight",
                             ", obsWeights = weight, shrinkage=.01", ", obsWeights = weight", ", obsWeights = weight") # I added weight to glmnet, bayesglm, gbm, lm and dbarts. svm, ksvm and randomForest don't seem to have a weighting option.


rmse_sam <- rep(0,length(model_list))
cor_sam <- rep(0,length(model_list))
rmse_test <- rep(0,length(model_list))
cor_test <- rep(0,length(model_list))
results <- data.frame(model_list, option_list_weight, rmse_sam, cor_sam, rmse_test, cor_test)


for(j in 1:length(model_list)){
  
  rmse <- rep(0,10)
  cor <- rep(0,10)
  pred_sample <- c(1:nrow(m_sample))
  p <- rep(1:10, each=nrow(m_sample)/10) # Define positions for the train+validation set to be divided into 10 pieces for 10-fold CV
  k <- j+26
  
  for(i in 1:10){
    train <- m_sample[which(p!=i),]
    label_train <- label_sample[which(p!=i)]
    weight_train <- weight[which(p!=i)]
    
    valid <- m_sample[which(p==i),]
    label_valid <- label_sample[which(p==i)]
    
    set.seed(150)  # set seed to reproduce the same model fit 
    model <- eval(parse(text=paste("SL.",model_list[j],"(label_train, train, valid, family = gaussian()",option_list_weight[j],")",sep="")))
    rmse[i] <- sqrt(mean((label_valid - model$pred)^2)) 
    cor[i] <- cor(label_valid, model$pred)
    
    pred_sample[which(p==i)] <- model$pred
  }
  results$rmse_sam[j] <- sum(rmse)/10   # For a j-th model
  results$cor_sam[j] <- sum(cor)/10   # For a j-th model
  
  
  pred_sample_file <- paste("pred_sample",k,".Rdata",sep="")
  save(pred_sample, file=pred_sample_file)
  
  
  ### Predict the test set of 300 paragraphs using the same model 
  model <- eval(parse(text=paste("SL.",model_list[j],"(label_sample, m_sample, m_test, family = gaussian()",option_list_weight_test[j],")",sep="")))
  results$rmse_test[j] <- sqrt(mean((label_test - model$pred)^2)) 
  results$cor_test[j] <- cor(label_test, model$pred)
  
  pred_test <- model$pred
  pred_test_file <- paste("pred_test",k,".Rdata",sep="")
  save(pred_test, file=pred_test_file)
  
  
  model_file <- paste("model",k,".Rdata",sep="")
  save(model, file=model_file)
  
  
  print(paste("done for processing for:", pred_test_file))
}

#save(results, file="results_tfidf_weight.Rdata")

