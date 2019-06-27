##### This is the code to fit prediction models using three types of document matrix
setwd("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec")
load("HouseHearings_submit.Rdata")  # load hearing data
load("grid.Rdata") # This is a product of "Rcode1_BuildDoc2vecMatrix"

library(SuperLearner)
#memory.limit(size=50000)


# Choose the data for training and testing: 
x <- which(data$dataset_new==1 | data$dataset_new==3) # select 3000 paragraphs = 1000 from the first run + 2000 from the third run
label <- data$y4 # y4 is the sentimentit score from the stan model fit on all the 3500 paragraphs
y <- length(x)/10 # define the number of test set to be set aside until the final analysis as one tenth of the data


# Divide the data into a train+validation set and a test set
set.seed(155)  
test_id <- sample(x,y) 
test_id <- sort(test_id, decreasing=FALSE)
sample_id <- x[which(!x %in% test_id)]


sample_id <- sample_id[sample(length(sample_id))] # shuffle the data
label_sample <- label[sample_id] # label for the sample data
label_test <- label[test_id] # get label for the test set


#save(label_sample, file="label_sample.Rdata")
#save(label_test, file="label_test.Rdata")
#save(sample_id, file="sample_id.Rdata")
#save(test_id, file="test_id.Rdata")


# Create entries where we will write rmse and cor measures for each doc2_sum matrix
grid$rmse_sam <- NA
grid$cor_sam <- NA


# For each of the 96 doc2_vec matrices, fit a ksvm model and compute 10-CV out-of-sample rmse and correlation between predictions and the labels
for(j in 1:nrow(grid)){
  filename <- paste("doc2_sum",j,".Rdata",sep="")
  load(filename)  # load each of the 96 doc2vec matrices from "doc2_sum1.Rdata" through "doc2_sum96.Rdata"


  m_sample <- as.data.frame(doc2_sum[sample_id,])  

  
  rmse <- rep(0,10)
  cor <- rep(0,10)
  p <- rep(1:10, each=nrow(m_sample)/10) # Define positions for the train+validation set to be divided into 10 pieces for 10-fold CV


  for(i in 1:10){
    train <- m_sample[which(p!=i),]
    label_train <- label_sample[which(p!=i)]
  
    valid <- m_sample[which(p==i),]
    label_valid <- label_sample[which(p==i)]
  
    set.seed(150)  # set seed to reproduce the same model fit 
    model <- SL.ksvm(label_train, train, valid, family = gaussian(), epsilon = 0.5) 

    rmse[i] <- sqrt(mean((label_valid - model$pred)^2)) 
    cor[i] <- cor(label_valid, model$pred)
  }
  grid$rmse_sam[j] <- sum(rmse)/10  
  grid$cor_sam[j] <- sum(cor)/10  

  rm(doc2_sum) 

  print(paste("done for processing for:", filename))
}

#save(grid, file="grid.Rdata")


newdata <- grid[order(-grid$cor_sam),] # by correlation (in a descending order)
head(newdata)

newdata <- grid[order(grid$rmse_sam),] # by RMSE (in an ascending order)
head(newdata)   # the output is slightly different from above using cor_sam, but the 62 is still the best


