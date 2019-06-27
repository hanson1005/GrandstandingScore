################################ Ensemble process #####################################
setwd("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec")
load("doc.features_2gram_2000.Rdata")
load("label_sample.Rdata")
load("label_test.Rdata")
load("sample_id.Rdata")
load("test_id.Rdata")


library(EBMAforecast)
library(SuperLearner)
library(quanteda)
library(text2vec)
memory.limit(size=50000)


### load sample predictions
for(i in 1:39){
  file_name <- paste0("pred_sample",i,".Rdata")
  load(file_name)
  if(i==1){
    allmodels <- pred_sample
  } else{
    allmodels <- cbind(allmodels, pred_sample)
  }
}

model_names <- c("svm", "svm2", "ksvm", "ksvm2", "glmnet", "glmnet2", "randomForest", "randomForest2", "bayesglm", "gbm", "gbm2", "lm", "dbarts")
colnames(allmodels) <- rep(model_names, 3)
colnames(allmodels) <- make.names(colnames(allmodels), unique = TRUE) # improve this naming. rather than adding ".1" or ".2", make them "_reg" and "_tfidf"
#save(allmodels, file="allmodels_3000_2gram_weight.Rdata")


### load predictions on the test set
for(i in 1:39){
  file_name <- paste0("pred_test",i,".Rdata")
  load(file_name)
  if(i==1){
    allmodels_test <- pred_test
  } else{
    allmodels_test <- cbind(allmodels_test, pred_test)
  }
}
colnames(allmodels_test) <- NULL
colnames(allmodels_test) <- rep(model_names, 3)
colnames(allmodels_test) <- make.names(colnames(allmodels_test), unique = TRUE) # run this before fitting gbm to avoid an error message
#save(allmodels_test, file="allmodels_test_3000_2gram_weight.Rdata")



### Fit an EBMA ############################################
fcobj <- makeForecastData(.predCalibration = allmodels,  # without test set info
                          .outcomeCalibration = label_sample,
                          .modelNames = colnames(allmodels))  # this just transforms the input data used to ensemble into a s4 class object which is acceptable for the ensemble command


ensemble <- calibrateEnsemble(fcobj, model = "normal")
summary(ensemble) 
# This doesn't change by excluding the tfidf models from allmodels 


w <- summary(ensemble)
weight <- w@summaryData[,1]
weight <- weight[-1]
final_models <- which(weight > 0)
final_models


#final_models
#ksvm   ksvm2  glmnet glmnet2   svm.1  ksvm.1 ksvm2.1  gbm2.1  gbm2.2 
#3       4       5       6      14      16      17      24      37 


#save(ensemble, file="ensemble_3000_2gram_weight.Rdata")
#save(final_models, file="final_models_3000_2gram_weight.Rdata")



### Predict the test set
pred_ebma <- EBMApredict(ensemble, allmodels_test, label_test)
EBMA_test <- pred_ebma@predTest[,,1][,1]   # access the 6 column matrix which is a cbind of EBMA prediction and allmodels_test


sqrt(mean((label_test - EBMA_test)^2))  # 0.6139154 
cor(label_test, EBMA_test) # 0.7029216  


plotdata <- as.data.frame(cbind(label_test, EBMA_test))
ebma_plot <- ggplot(plotdata, aes(label_test, EBMA_test)) +
  geom_point() +
  theme_bw() +
  labs(x="SentimentIt Score of Grandstanding", y = "Predicted Grandstanding Score")
#save(ebma_plot, file="ebma_graph_3000_2gram_weight.Rdata")



### Predict the rest of the corpus
# The following code retrieves the model numbers that received non-zero weight. Create a matrix with dimension of nrow(data) by 39 filled with zeros.
# Fill the columns for the models with non-zero weight, by predicting the values. 
# Then, I predict the corpus using each of the models with non-zero weight. predict.SL.XXX works for most of the models chosen (e.g. svm, ksvm, glmnet, randomForest, gbm). 
# Alternatively, I can also do "newX=m" when fitting a model. However, randomForest and gbm give error message of "cannot allocate the vector size of X.X GB in memory". 
# R session sometimes gets aborted or does not generate a vector of predicted values. 
# This problem exists even when fitting the model with newX=m_test and then use predict.SL.XXX. 
# As a result, I had to divide up the matrix into half or even to more pieces and use "predict.SL.XXX" for each piece.
# Although the following code uses for loop to go through all 9 final models in a one shot, but in practice,
# this is hard to be done in one click because multiple document matrices should be loaded one at a time.
# It is not impossible but many can encounter a memory error.


### Looping over final models
load("pred_corpus_3000_2gram_weight.Rdata")
pred_corpus <- matrix(NA, nrow(data), length(final_models))
model_list <- c("svm", "svm", "ksvm", "ksvm", "glmnet", "glmnet", "randomForest", "randomForest", "bayesglm", "gbm", "gbm", "lm", "dbarts")

set.seed(150)
for(i in 1:length(final_models)){       # load a saved model
  j <- final_models[i]
  
  if(j==min(final_models[which(final_models<=length(model_list))])){
    load("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec/doc2_sum62.Rdata") 
    m <- as.data.frame(doc2_sum)  
    m1 <- m[1:600000,]
    m2 <- m[600001:1029427,]   # divide the corpus into half to avoid a memory error when generating a vector of predicted values
    rm(doc2_sum)
    gc()
    print("From here, m = doc2_vec")
    
    k=j
    
  } else if(j==min(final_models[which(final_models>length(model_list) & final_models<=2*length(model_list))])){
    load("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec/doc.features_2gram_2000.Rdata")
    m <- as.matrix(doc.features)
    m <- as.data.frame(m)
    m1 <- m[1:600000,]
    m2 <- m[600001:1029427,]   
    print("From here, m = regular document matrix")
    
    k=j-13
    
  } else if(j==min(final_models[which(final_models>2*length(model_list))])){
    tfidf = TfIdf$new()
    doc.features_tfidf = fit_transform(doc.features, tfidf)
    m <- as.matrix(doc.features_tfidf)
    m <- data.frame(m)
    m1 <- m[1:600000,]
    m2 <- m[600001:1029427,]   
    rm(doc.features, doc.features_tfidf)
    gc()
    print("From here, m = tfidf document matrix")
    
    k=j-26
    
  } else {}
  
  if(k==13){
    m_sample <- m[sample_id,]  
    
    model <- SL.dbarts(label_sample, m_sample, m1, family = gaussian())
    pred_corpus[1:nrow(m1),i] <- model$pred
    
    model <- SL.dbarts(label_sample, m_sample, m2, family = gaussian())
    pred_corpus[(nrow(m1)+1):nrow(m),i] <- model$pred
    
  } else{
    model_file <- paste("model",j,".Rdata",sep="")
    load(model_file)
    
    pred_corpus[1:nrow(m1),i] <- eval(parse(text=paste0("predict.SL.",model_list[k],"(model$fit, m1 ,family = gaussian())")))   # predict the corpus
    pred_corpus[(nrow(m1)+1):nrow(m),i] <- eval(parse(text=paste0("predict.SL.",model_list[k],"(model$fit, m2, family = gaussian())")))  
  }  
  
  print(paste("done for processing for:", model_file))
  save(pred_corpus, file="pred_corpus_3000_2gram_weight.Rdata")
}




# create a matrix of priors for the corpus
allmodels_corpus <- matrix(0, nrow(data), ncol(allmodels))
colnames(allmodels_corpus) <- colnames(allmodels)


for(i in 1:length(final_models)){
  model_num <- final_models[i]
  allmodels_corpus[,model_num] <- pred_corpus[,i]
}
#save(allmodels_corpus, file="allmodels_corpus.Rdata")




# make ebma predictions
d1 <- allmodels_corpus[1:nrow(m1),]
d2 <- allmodels_corpus[(nrow(m1)+1):nrow(m),]

y3000_ebma <- NA

pred_ebma <- EBMApredict(ensemble, d1)
y3000_ebma <- pred_ebma@predTest[,,1][,1]
save(y3000_ebma, file="y3000_ebma_2gram.Rdata")

pred_ebma <- EBMApredict(ensemble, d2)
a <- pred_ebma@predTest[,,1][,1]
y3000_ebma <- c(y3000_ebma, a)
save(y3000_ebma, file="y3000_ebma_2gram.Rdata")


data$y3000_ebma_2gram <- y3000_ebma
#save(data, file="HouseHearings_submit.Rdata")



### Combine 10-fold CV results from all three document matrices
load("results_doc2vec_weight.Rdata")
results_d2v <- results
results_d2v$matrix <- "Doc2vec"

load("results_reg_weight.Rdata")
results_reg <- results
results_reg$matrix <- "Bag of words"

load("results_tfidf_weight.Rdata")
results_tfidf <- results
results_tfidf$matrix <- "Tf-idf"

results <- rbind(results_d2v, results_reg, results_tfidf)
mlist <- c("svm", "svm2", "ksvm", "ksvm2", "glmnet", "glmnet2", "randomForest", "randomForest2", "bayesglm", "gbm", "gbm2", "lm", "dbarts")
results$model_list <- rep(mlist, 3)
results$model_num <- rep(1:13,3)
results$model_num <- as.factor(results$model_num)
#save(results, file = "results_combined.Rdata")

