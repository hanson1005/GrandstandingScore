##### This is the code to construct doc2vec matrix
setwd("~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec")
load("HouseHearings_submit.Rdata")  # load hearing data
#memory.limit(size=50000)

#install.packages("devtools")
#install.packages("textTinyR")    # install relevant packages
devtools::install_github('mlampros/fastTextR')
library(textTinyR)
library(fastTextR)
library(stopwords)
stopwordslist <- stopwords(language = "en", source = "snowball")
stopwords_more <- c("Im", "youre", "hes", "shes", "its", "were", "theyre", "ive", "youve", "weve", "theyve", "id", "youd", "hed", "shed",
                    "wed", "theyd", "ill", "youll", "hell", "shell", "well", "theyll", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
                    "hadnt", "doesnt", "dont", "didnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt", 
                    "lets", "thats", "whos", "whats", "heres", "theres", "whens", "wheres", "whys", "hows", "aint", "ain't") 
stopwordslist <- c(stopwordslist, stopwords_more)



##### 1) Retrieve corpus from data and tokenize it
string <- data$speech
concat <- gsub("--|---|----", " ", string)  # remove "--" or "----" from speeches to avoid creating non-existing words (e.g. you--you => youyou)


### RUN THIS ONLY ONCE!
# Tokenize the corpus and save it as "output_token_single_file.txt" file in the working directory (don't have to run this code again)
save_dat = textTinyR::tokenize_transform_vec_docs(object = concat, as_token = T, 
                                                  to_lower = T, 
                                                  remove_punctuation_vector = T, 
                                                  remove_numbers = F, trim_token = T, 
                                                  split_string = T, 
                                                  split_separator = " \r\n\t.,;:()?!//",
                                                  remove_stopwords = stopwordslist, 
                                                  stemmer = "porter2_stemmer", 
                                                  path_2folder = "~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec/", 
                                                  threads = 1, verbose = T)


# Tokenize the corpus again (do this again because the previous output was saved in the designated folder in .txt format and not loaded in the R session.
# In order to feed the tokens to fasttext, it has to be in a folder, but in order to construct doc2vec using textTinyR, it has to be loaded in the R session.) 
clust_vec = textTinyR::tokenize_transform_vec_docs(object = concat, as_token = T,
                                                   to_lower = T, 
                                                   remove_punctuation_vector = T,  # it was F originally
                                                   remove_numbers = F, trim_token = T,
                                                   split_string = T,
                                                   split_separator = " \r\n\t.,;:()?!//", 
                                                   remove_stopwords = stopwordslist, 
                                                   stemmer = "porter2_stemmer", 
                                                   threads = 1, verbose = T)




##### 2) Fit a model to create word vectors ("rt_fst_model.vec")  & "rt_fst_model.bin" using Facebook's fasttext which is available through "fastTextR" R package
PATH_INPUT = "C:/Users/juyeon/Desktop/New folder/output_token_single_file.txt"
PATH_OUT = "C:/Users/juyeon/Desktop/New folder/rt_fst_model"


dim <- c(300, 400, 600)
ws <- c(5, 7, 10, 12)
epoch <- c(10, 15, 30, 45)
wordNgram <- c(1, 2)


grid <- expand.grid(dim=dim, ws=ws, epoch=epoch, wordNgram=wordNgram)
#save(grid, file="grid.Rdata")


for(i in 1:nrow(grid)){   # Although this is in a for loop code, I ran it at CHPC using a batch model since this loop will take 5-7 days on one computer even with the thread set at 8.
vecs = fastTextR::skipgram_cbow(input_path = PATH_INPUT, output_path = PATH_OUT, 
                                method = "skipgram", lr = 0.02, lrUpdateRate = 100, 
                                dim = grid[i,1], ws = grid[i,2], epoch = grid[i,3], minCount = 1, neg = 5, 
                                wordNgrams = grid[i,4], loss = "ns", bucket = 2e+06,   # If loss = "hs" instead of "ns", it takes less time to fit a model, but I haven't tried this yet (https://github.com/facebookresearch/fastText/issues/507)
                                minn = 0, maxn = 0, thread = 8, t = 1e-04, verbose = 2)  # Although I am using fasttext but since minn and maxn are set at zero, it is like I am fitting a word2vec model.  


##### 3) Create doc2vec file 
# build a document matrix using the tokenized corpus and word vectors in "rt_fst_model.vec"
init = textTinyR::Doc2Vec$new(token_list = clust_vec$token, 
                              word_vector_FILE = "~/Dropbox/Hearing_SentimentIt/EBMA/doc2vec/rt_fst_model.vec",
                              print_every_rows = 5000, 
                              verbose = TRUE, 
                              copy_data = FALSE)                 


doc2_sum = init$doc2vec_methods(method = "sum_sqrt", threads = 6)
filename <- paste("doc2_sum",i,".Rdata",sep="")
save(doc2_sum, file=filename)

print(paste("done for processing for:", filename))

}


