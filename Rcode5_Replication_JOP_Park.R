setwd("C:/Users/juyeon/Desktop/New Projects/Hearing Text/Publication/JOP/Replication_HearingText")
library(lmerTest) 
library(lmtest)
library(lme4)
library(plm) 
library(MatchIt)
options(scipen=999)
options(max.print=1000000)



load("Data_member.Rdata")
load("Data_speech.Rdata")



###################################### Tables ###########################################
##### Table 2 on Regression Results
### Model 1
m1 <- lm(gscore ~ as.factor(govtrack) + minority + unified + minuni + chair + subchr + leader + talk_rs + as.factor(congress), data=member) 
nobs(m1) 
coeftest(m1, vcov=vcovHC(m1,type="HC0",cluster="govtrack")) 


### Model 2
m2 <- lm(gscore ~ as.factor(govtrack) + minority + unified + minuni + partyloyalty + votepct100 + votepct_sq100 + seniority_rs + seniority_sq_rs
         + abs_dwnom1_rs + dem + freshman + female + chair + subchr + leader + talk_rs + as.factor(congress), data=member) 
nobs(m2) 
coeftest(m2, vcov=vcovHC(m2,type="HC0",cluster="govtrack")) 


### Model 3
speech <- within(speech, committee_code2 <- relevel(committee_code2, ref = "HSGO"))
m3 <- lmer(gscore ~ minority + unified + minuni + chairspeech + rankmemspeech + leader + salience_rs + polar_rs 
           + as.factor(committee_code2) + as.factor(congress) + (1| govtrack) +(1 | file_name), data=speech)
nobs(m3)
summary(m3) 


### Model 4
m4 <- lmer(gscore ~ minority + unified + minuni + partyloyalty + votepct100 + votepct_sq100 + seniority_rs + seniority_sq_rs  
            + abs_dwnom1_rs + dem + freshman + female + chairspeech + rankmemspeech + leader + salience_rs + polar_rs 
            + as.factor(committee_code2) + as.factor(congress) + (1| govtrack) +(1 | file_name), data=speech)
nobs(m4)  
summary(m4) 





##### Table 3 on Matching
### Powerful Committees
speech_nonchair <- speech[speech$chairspeech==0,] # Exclude chairs
match <- aggregate(gscore ~ congress + govtrack + powercmt, speech_nonchair, FUN=mean) # Aggregate by legislator & congress & powerful committee 
m.out <- matchit(powercmt ~ congress + govtrack, data = match, method = "exact")
m.out # summary of matched pairs
m.data_power <- match.data(m.out) # generate a matched data set
t.test(gscore ~ powercmt, data=m.data_power, conf.level=0.95)  # Difference in Means (Without chairs)


##### Foreign and Security-related Committees
match <- aggregate(gscore ~ congress + govtrack + security, speech, FUN=mean) # Aggregate by legislator & congress & security  
m.out <- matchit(security ~ congress + govtrack, data = match, method = "exact")
m.out 
m.data_security <- match.data(m.out)
t.test(gscore ~ security, data=m.data_security, conf.level=0.95) # Difference in Means  





###################################### Figures ###########################################
### Figure 1
load("Data_training.Rdata")
label <- training_data$sentimentit_score

par(mfrow=c(1,3))
hist(label, main="Sample Speeches", xlab="Grandstanding Score")
hist(speech$gscore, main="Statement-level Data",  xlab="Grandstanding Score (Rescaled)")
hist(member$gscore, main="Member-level Data",  xlab="Grandstanding Score (Rescaled)")
 


### Figure 2
library(gplots)
par(mar=c(5,5,0,1))
figure2 <- plotmeans(gscore ~ year, data = speech, frame = FALSE, ylab = "Average Grandstanding Score", xlab = "Years of Congress", n.label=FALSE)
figure2



### Figure 3
library(dplyr)
result <- data.frame(matrix(, nrow = 10, ncol = 2))
colnames(result) <- c("lower", "upper")

for (i in 105:114){
  df <- speech %>%
    filter(congress==i) %>%
    select(gscore, dem)
  
  a <- t.test(gscore ~ dem, data = df)
  #summary(a)
  j <- i-104
  result[j,] <- a$conf.int
}


result$est <- (result$upper+result$lower)/2
result$congress <- as.factor(c(105:114))
result$rep_minority <- ifelse(result$congress==110|result$congress==111, 1, 0)


library(ggplot2)
(p1 <- ggplot(result, aes(congress, est))+
    geom_point()+
    geom_hline(yintercept=0) +
    geom_errorbar(aes(ymin=lower, ymax=upper, color=factor(rep_minority)), width=.1) +
    theme_bw() +
    xlab("Congress") + ylab("Mean Difference (Republicans - Democrats)") +
    scale_colour_discrete(name  ="Minority Party",
                          breaks=c("0", "1"),
                          labels=c("Democrat", "Republican")))



### Figure 4: Grandstanding Score by DW-Nominate Scores for the 114th Congress
library(ggrepel)
upper <- quantile(member$gscore, .75) + 1.5*(quantile(member$gscore, .75)-quantile(member$gscore, .25)) # Define outliers to be labeled using +/- 1.5*IQR
lower <- quantile(member$gscore, .25) - 1.5*(quantile(member$gscore, .75)-quantile(member$gscore, .25))
member114 <- member[which(member$congress==114),] # Select only the 114th 
member114$unreliable <- as.factor(ifelse(member114$talk <= 10 , 1, 0)) # "talk" variable counts the number of speeches each legislator made in each Congress
figure4 <- ggplot(member114, aes(gscore, dwnom1, color=unreliable)) +
  scale_color_manual(values=c("#000000", "#696969")) +
  geom_point() +
  guides(fill=FALSE, color=FALSE) +
  theme_bw() +
  geom_text_repel(data = subset(member114, gscore > upper | gscore < lower), aes(label = thomas_name))+
  labs(x="Propensity to Grandstand", y = "DW-Nominate (Dimension1)")
figure4



### Figure 5
beta <- fixef(m4)
se <- sqrt(diag(vcov(m4, useScale = FALSE)))
cmtfix <- as.data.frame(cbind(beta, se))[19:39,]
cmtfix$cmtname <- c("Intelligence (Select)", "Benghazi (Select)", "Agriculture", "Appropriations", "Armed Services", 
                     "Financial Services", "Budget", "Education and the Work Force", "Foreign Affairs", 
                     "Energy Independence and Global Warming (Select)", "House Administrations","Homeland Security", "Energy and Commerce",
                     "Natural Resources", "Judiciary", "Transportation and Infrastructure", "Rules", "Small Business and Entrepreneurship", "Science, Space, and Technology",
                     "Veterans' Affairs", "Ways and Means")
cmtfix$type <- 0
cmtfix$type[cmtfix$cmtname %in% c("Budget", "Rules", "Appropriations", "Ways and Means")] <- 1
cmtfix$type[cmtfix$cmtname %in% c("Intelligence (Select)", "Armed Services", "Foreign Affairs", "Benghazi (Select)")] <- 2
table(cmtfix$type)

cmtfix$u_ci <- cmtfix$beta + 1.96*cmtfix$se
cmtfix$l_ci <- cmtfix$beta - 1.96*cmtfix$se 
cmtfix$cmtname <- factor(cmtfix$cmtname, levels = cmtfix$cmtname[order(cmtfix$beta)])


figure5 <- ggplot(cmtfix, aes(x = cmtname, y = beta, label=cmtname)) + geom_point() +labs(x = "Committees", y="Coefficient") +
  geom_errorbar(aes(ymin=u_ci, ymax=l_ci), width=.1) +
  geom_text(aes(color=factor(type)), size=4, angle=90, hjust=.2, vjust=-.5) +
  scale_colour_manual(name = element_blank(), values=c("grey60", "orange", "purple"), labels = c("Other", "Powerful", "Foreign-Security")) +
  geom_hline(yintercept = 0) + 
  theme(panel.background = element_blank(), axis.text.x=element_blank(),
        axis.ticks.x=element_blank(), legend.position = "none")
figure5





###################################### Appendix (for Print) ##########################################
##### Top and bottom 30 speeches
topspeech <- speech[,c("speech", "gscore")]
topspeech <- topspeech[order(-topspeech[,2]),]
topspeech$speech[1:30] # Top 30


bottom <- topspeech$speech[(nrow(speech)-30):nrow(speech)]  
bottom <- bottom[-11] # Remove the 11th statement since it contains only two words spoken
bottom # Bottom 30





##### 200 most frequent (stemmed) words in grandstanding and non-grandstanding statements
library(quanteda)
library(SuperLearner)
library(stopwords)


stopwordslist <- stopwords(language = "en", source = "snowball")
stopwords_more <- c("Im", "youre", "hes", "shes", "its", "were", "theyre", "ive", "youve", "weve", "theyve", "id", "youd", "hed", "shed",
                    "wed", "theyd", "ill", "youll", "hell", "shell", "well", "theyll", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
                    "hadnt", "doesnt", "dont", "didnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt", 
                    "lets", "thats", "whos", "whats", "heres", "theres", "whens", "wheres", "whys", "hows", "aint", "ain't") 
stopwordslist <- c(stopwordslist, stopwords_more)
string <- speech$speech
string <- gsub("--|---|----", " ", string)  # remove "--", "---" or "----" from speeches to avoid creating non-existing words (e.g. you--you => youyou)


# Create corpus and document feature matrix (dfm)
corpus <- corpus(string, docvars=speech) 
doc.features <- dfm(corpus, remove =stopwordslist, stem=TRUE, remove_punct = TRUE) 
doc.features <- dfm_trim(doc.features, min_termfreq =2000, max_termfreq= 51, termfreq_type = "rank") # exclude the most frequent 50 words in the entire corpus


# Create a style variable using thresholds based on quartiles 
qt <- quantile(speech$gscore, c(.25, .75))
speech$grand = ifelse((speech$gscore > qt[2] ), 1, 0)
speech$non_grand = ifelse((speech$gscore < qt[1]), 1, 0)
speech$style = speech$grand-speech$non_grand  # 1 if grandstanding; -1 if info-seeking, and 0 otherwise


# List the 200 most frequent words for each of the two styles
grandstanding <- names(topfeatures(doc.features[which(speech$grand==1),], n = 200, decreasing = TRUE, scheme = c("count", "docfreq")))
grandstanding

non_grand <- names(topfeatures(doc.features[which(speech$non_grand==1),], n = 200, decreasing = TRUE, scheme = c("count", "docfreq")))
non_grand


# Identify the words overlapping in both lists
overlap <- grandstanding[(grandstanding %in% non_grand)]
overlap # 117 words


# Measure the relative importance of the overlapping words in characterizing grandstandings statements
overlap <- as.data.frame(overlap)  
overlap$gapinorder <- NA 
for (i in 1:nrow(overlap)){
  overlap$gapinorder[i] <- which(non_grand==overlap[i,1]) - which(grandstanding==overlap[i,1])
}
overlap <- overlap[order(-overlap$gapinorder),]
View(overlap)





################################### Online Supporting Materials ######################################
############################################## Figures ################################################
### Summary of results from the 39 machine learning models
# Figure A1  
load("Summary_of_39learners.Rdata")
ggplot(learners, aes(x=model_num, y=cor_sam, color=matrix, group=matrix)) +
  geom_point() +
  geom_line() +
  theme_bw() +
  theme(axis.text.x = element_text(angle=45, hjust = 1)) +
  scale_x_discrete(breaks=c(1:13), labels=c("svm", "svm2", "ksvm", "ksvm2", "glmnet", "glmnet2", "randomForest", "randomForest2", "bayesglm", "gbm", "gbm2", "lm", "dbarts")) +
  labs(x="Models", y = "Correlation Between Labels and Predictions") +
  labs(color = "Document Matrix")



# Figure A2
ggplot(learners, aes(x=model_num, y=rmse_sam, color=matrix, group=matrix)) +
  geom_point() +
  geom_line() +
  theme_bw() +
  theme(axis.text.x = element_text(angle=45, hjust = 1)) +
  scale_x_discrete(breaks=c(1:13), labels=c("svm", "svm2", "ksvm", "ksvm2", "glmnet", "glmnet2", "randomForest", "randomForest2", "bayesglm", "gbm", "gbm2", "lm", "dbarts")) +
  labs(x="Models", y = "RMSE") +
  labs(color = "Document Matrix")



### Sensitivity of grandstanding score to sample size
# Figure A3: Member-level sensitivity
d1000 <- member[member$talk >=1000,] # Subset only those who spoke more than 1000 times
d1000$mid <- paste0(d1000$congress, d1000$govtrack)
speech$mid <- paste0(speech$congress, speech$govtrack)

d_speech <- speech[speech$mid %in% d1000$mid,]
size <- seq(10, 1000, by=10)
mat <- matrix(nrow = length(size), ncol = 10) # a matrix of average grandstanding score for each d1000$mid by sample size and 10 iterations 
bigmat <- matrix(nrow = length(size), ncol = nrow(d1000))

set.seed(150)
for (k in 1:nrow(d1000)){
  for (i in 1:length(size)){
    for (j in 1:10){
      a <- d_speech[sample(nrow(d_speech), size[i]),]
      mat[i,j] <- mean(a$gscore)
    }
    bigmat[i,k] <- sd(mat[i, 1:10]) # get the sd of an individual in a congress for sample size `i' across ten iterations
  }
}

bigmat2 <- bigmat^2
finalvalue <- NA
for (i in 1:length(size)){
  finalvalue[i] <- sqrt(sum(bigmat2[i,])/nrow(d1000)) # compute the pooled sd for each sample size
}

finalvalue <- as.data.frame(finalvalue)
finalvalue$size <- size

ggplot(finalvalue, aes(x=size, y=finalvalue)) + geom_point() +   # draw a plot
  labs(x="Sample Size", y = "Pooled Standard Deviation") +
  theme_bw() 



# Figure A4: Congress-level sensitivity 
d_cong <- speech[speech$congress==110,] # Random sampling from the 110th Congress which has the most observations
size <- c(30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)
mat2 <- matrix(nrow = length(size), ncol = 11)

set.seed(150)
for (i in 1:length(size)){
  for (j in 1:10){
    a <- d_cong[sample(nrow(d_cong), size[i]),]
    mat2[i,j] <- mean(a$gscore)
  }
  mat2[i, 11] <-mean(mat2[i,1:10])
}

mat2 <- as.data.frame(mat2)
mat2$size <- size

ggplot(mat2, aes(x=size, y=V11)) + geom_point() +
  labs(x="Sample Size", y = "Standard Deviation") +
  theme_bw() +
  scale_x_continuous(breaks = c(0,40000,60000, 80000, 10000))



# Figure A5: Committee-level sensitivity  
d_cmt <- speech[speech$committee_code2=="HSGO",] # Random sampling from the HSGO which has the most observations
size <- c(500, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 20000, 30000, 40000, 50000, 80000, 100000)
mat3 <- matrix(nrow = length(size), ncol = 11)

set.seed(150)
for (i in 1:length(size)){
  for (j in 1:10){
    a <- d_cmt[sample(nrow(d_cmt), size[i]),]
    mat3[i,j] <- mean(a$gscore)
  }
  mat3[i, 11] <-mean(mat3[i,1:10])
}

mat3 <- as.data.frame(mat3)
mat3$size <- size

ggplot(mat3, aes(x=size, y=V11)) + geom_point() +
  labs(x="Sample Size", y = "Standard Deviation") +
  theme_bw() 





############################################## Tables ################################################
# Table A2: Number of observations by Congress
library(plyr)
count <- count(member,c('congress'))  
colnames(count)[2] <- "member_level"
count$statement_level <- count(speech,c('congress'))[,2]  
count



# Table A3: Descriptive Statistics
library(psych)
des_stat <- member %>%
  dplyr::select(.data$gscore, .data$minority, .data$unified, .data$partyloyalty, .data$votepct, .data$votepct100,
                .data$seniority, .data$seniority_rs, .data$abs_dwnom1, .data$abs_dwnom1_rs, .data$dem, .data$freshman,
                .data$female, .data$chair, .data$subchr, .data$leader, .data$talk, .data$talk_rs) %>%
  describe() %>%
  as.data.frame()
des_stat <- des_stat[,c(8,3,9,4)]
rownames(des_stat) <- c("Grandstanding(Member-level)", "Minority", "Unified", "Party Support", "Vote(%)", "Vote(%)_rescaled",
                         "Seniority", "Seniority_rescaled", "Abs(DW-Nom.)", "Abs(DW-Nom.)_rescaled", "Democrat", "Freshman",
                         "Female", "Chair", "Subcommittee Chair", "Party Leadership", "Statement Frequency", "Statement Frequency_rescaled")
                         

des_stat2 <- speech %>%
  dplyr::select(.data$gscore, .data$chairspeech, .data$rankmemspeech, .data$salience, .data$salience_rs, .data$polar, .data$polar_rs)  %>%
  describe() %>%
  as.data.frame()
des_stat2 <- des_stat2[,c(8,3,9,4)]
rownames(des_stat2) <- c("Grandstanding(Statement-level)", "Chair's Statement", "Ranking Member's Statement", 
                         "Number of Speakers", "Number of Speakers_rescaled", "Polarization within Committee", "Polarization within Committee_rescaled")


des_stat <- round(rbind(des_stat, des_stat2),3)
des_stat <- des_stat[c("Grandstanding(Member-level)", "Grandstanding(Statement-level)", "Minority", "Unified", "Party Support", "Vote(%)", "Vote(%)_rescaled",
                        "Seniority", "Seniority_rescaled", "Abs(DW-Nom.)", "Abs(DW-Nom.)_rescaled", "Democrat", "Freshman",
                        "Female", "Chair", "Subcommittee Chair", "Chair's Statement", "Ranking Member's Statement", 
                        "Party Leadership", "Statement Frequency", "Statement Frequency_rescaled",
                        "Number of Speakers", "Number of Speakers_rescaled", "Polarization within Committee", "Polarization within Committee_rescaled"),]
des_stat


### Computed pooled variance of the member-level standard deviation across time
library(dplyr)
psd <- member%>%group_by(govtrack)%>%dplyr::summarise(Variance=var(gscore))
psd$n <- aggregate(gscore ~ govtrack, data = member, FUN = length)[,2]
psd <- psd[which(psd$n>1),] # drop cases with only one observation because their variances are "NA"
psd$n_1 <- psd$n-1
psd$n_1v <- psd$n_1*psd$Variance
sqrt(sum(psd$n_1v)/sum(psd$n_1))  # 4.33684





# Note: Code for Table A4 is available following the code for Table A6






####### Robustness checks
##### Table A5 Regression Results Using Matched Datasets
### Power Committees
# Model 5
member_temp <- member
member_temp$gscore <- NULL
md1 <- merge(m.data_power, member_temp, by=c("govtrack", "congress"), all.x=TRUE)
m5 <- lm(gscore ~ as.factor(govtrack) + powercmt + minority + unified + minuni + leader + talk_rs + as.factor(congress), data=md1) 
nobs(m5) 
coeftest(m5, vcov=vcovHC(m5,type="HC0",cluster="govtrack")) 


# Model 6
m6 <- lm(gscore ~ as.factor(govtrack) + powercmt + minority + unified + minuni + partyloyalty + votepct100 + votepct_sq100 + seniority_rs + seniority_sq_rs
            + abs_dwnom1_rs + dem + freshman + female + leader + talk_rs + as.factor(congress), data=md1) 
nobs(m6) 
coeftest(m6, vcov=vcovHC(m6,type="HC0",cluster="govtrack")) 


### Foreign and Security Committees
### Model 7
md2 <- merge(m.data_security, member_temp, by=c("govtrack", "congress"), all.x=TRUE)
rm(member_temp)
m7 <- lm(gscore~ as.factor(govtrack) + security + minority + unified + minuni + chair + subchr + leader + talk_rs + as.factor(congress), data=md2) 
nobs(m7) 
coeftest(m7, vcov=vcovHC(m7,type="HC0",cluster="govtrack")) 


### Model 8
m8 <- lm(gscore~ as.factor(govtrack) + security + minority + unified + minuni + partyloyalty + votepct100 + votepct_sq100 + seniority_rs + seniority_sq_rs
            + abs_dwnom1_rs + dem + freshman + female + chair + subchr + leader + talk_rs + as.factor(congress), data=md2) 
nobs(m8) 
coeftest(m8, vcov=vcovHC(m8,type="HC0",cluster="govtrack")) 






# Table A6 Regression Results without Select Committees
load("Data_member_ns.Rdata")
load("Data_speech_ns.Rdata")


# Model 9
m9 <- lm(gscore~ as.factor(govtrack) + minority + unified + minuni + chair + subchr + leader + talk_rs + as.factor(congress), data=member_ns) 
nobs(m9) 
coeftest(m9, vcov=vcovHC(m9,type="HC0",cluster="govtrack")) 


# Model 10
m10 <- lm(gscore~ as.factor(govtrack) + minority + unified + minuni + partyloyalty + votepct100 + votepct_sq100 + seniority_rs + seniority_sq_rs
          + abs_dwnom1_rs + dem + freshman + female + chair + subchr + leader + talk_rs + as.factor(congress), data=member_ns) 
nobs(m10) 
coeftest(m10, vcov=vcovHC(m10,type="HC0",cluster="govtrack"))


# Model 11
speech_ns <- within(speech_ns, committee_code2 <- relevel(committee_code2, ref = "HSGO"))
m11 <- lmer(gscore~ minority + unified + minuni + chairspeech + rankmemspeech + leader + salience_rs + polar_rs + as.factor(committee_code2) + as.factor(congress)
            + (1| govtrack) +(1 | file_name), data=speech_ns)
nobs(m11)
summary(m11) 


# Model 12
m12 <- lmer(gscore~ minority + unified + minuni + partyloyalty + votepct100 + votepct_sq100 + seniority_rs + seniority_sq_rs  
             + abs_dwnom1_rs + dem + freshman + female + chairspeech + rankmemspeech + leader + salience_rs + polar_rs 
             + as.factor(committee_code2) + as.factor(congress) + (1| govtrack) +(1 | file_name), data=speech_ns)
nobs(m12)
summary(m12) 





# Table A4
beta <- fixef(m12)
se <- sqrt(diag(vcov(m12, useScale = FALSE)))
cmtfix2 <- as.data.frame(cbind(beta, se))[19:36,]
cmtfix2$cmtname <- c("Agriculture", "Appropriations", "Armed Services", 
                       "Financial Services", "Budget", "Education and the Work Force", "Foreign Affairs", 
                       "House Administrations","Homeland Security", "Energy and Commerce",
                       "Natural Resources", "Judiciary", "Transportation and Infrastructure", "Rules", "Small Business and Entrepreneurship", 
                       "Science, Space, and Technology", "Veterans' Affairs", "Ways and Means")
cmtfix2$coef2 <- paste0(round(cmtfix2$beta,3),"(", round(cmtfix2$se,3), ")")
cmtfix$coef <- paste0(round(cmtfix$beta,3),"(", round(cmtfix$se,3), ")")


cmt <- aggregate(speech[, c("gscore")], list(speech$committee_code2), mean)
cmt <- cmt[order(cmt[,1]),]
cmt$cmtname <- c("Oversight and Government Reform", "Intelligence (Select)", "Benghazi (Select)", "Agriculture", "Appropriations", "Armed Services", 
                "Financial Services", "Budget", "Education and the Work Force", "Foreign Affairs", "Energy Independence and Global Warming (Select)", 
                "House Administrations","Homeland Security", "Energy and Commerce",
                "Natural Resources", "Judiciary", "Transportation and Infrastructure", "Rules", "Small Business and Entrepreneurship", 
                "Science, Space, and Technology", "Veterans' Affairs", "Ways and Means")

cmt <- merge(cmt[,2:3], cmtfix[,c("coef", "cmtname")], by="cmtname", all.x=TRUE)
cmt <- merge(cmt, cmtfix2[,c("coef2", "cmtname")], by="cmtname", all.x=TRUE)
cmt$x <- round(cmt$x,3)
colnames(cmt) <- c("Committee", "Mean", "Coefficients from Model 4", "Coefficients from Model 12")
cmt <- cmt[order(-cmt[,2]),]
View(cmt)  





# Table A7
### Powerful committee causal effect
speech_nonchair <- speech_ns[speech_ns$chairspeech==0,] # exclude chairs
match <- aggregate(gscore ~ congress + govtrack + powercmt, speech_nonchair, FUN=mean) # Aggregate by legislator & congress & powerful committee 
m.out <- matchit(powercmt ~ congress + govtrack, data = match, method = "exact")
m.data <- match.data(m.out) # generate a matched data set
t.test(gscore ~ powercmt, data=m.data, conf.level=0.95) # Difference in Means


### Foreign and security-related committees causal effect
match <- aggregate(gscore ~ congress + govtrack + security, speech_ns, FUN=mean) 
m.out <- matchit(security ~ congress + govtrack, data = match, method = "exact")
m.data <- match.data(m.out)
t.test(gscore ~ security, data=m.data, conf.level=0.95) # Difference in Means

