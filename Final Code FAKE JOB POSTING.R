getwd()

setwd("C:/Users/pooja/Downloads/STUDY/Sem-2/Data Analytics/Project/fake_job_postings.csv")

#Packages used

install.packages("tm")
install.packages("PASWR2")
install.packages("caret")
install.packages("Metrics",dependencies=TRUE)
install.packages("e1071")
install.packages("ipred")
install.packages("class")
install.packages("stringr")
install.packages("SnowballC")
install.packages("dummies")

#Reading Data
job_posting = read.csv("fake_job_postings.csv",header=T)
head(job_posting)
nrow(job_posting)

#Data Preprocessing

#01_Elimination of columns that would not be used

#Dropping columns job_id, title, location, department, industry and function.

job_posting=job_posting[,-c(1,2,3,4,16,17)]
colnames(job_posting)

#02_Replacing the blank values with NA and then omitting the rows with NA value for column 'employment_type'

job_posting[job_posting==""] <- NA
sum(is.na(job_posting$employment_type))
na.omit(job_posting$employment_type)

sum(is.na(job_posting$description))

#03_Converting nominal variable 'employment_type' to binary variable
#Creating N-1 dummy variables
library(dummies)

job_posting=dummy.data.frame(job_posting,names=c("employment_type"))
colnames(job_posting)
head(job_posting,3)
#To drop one variable so that we have N-1 dummy variables
job_posting=job_posting[,-c(14)]

#04_Assigning column names to variables

x1 <- job_posting$description
x2 <- job_posting$telecommuting
x3 <- job_posting$has_company_logo
x4 <-job_posting$has_questions
x5 <- job_posting$employment_typeContract
x6 <- job_posting$`employment_typeFull-time`
x7 <- job_posting$employment_typeOther
x8 <- job_posting$`employment_typePart-time`
x9 <- job_posting$employment_typeTemporary
x10 <- job_posting$fraudulent

#Two-Sample Hypothesis Testing

#We are considering columns 'has_questions' and 'has_company_logo' for 2 sample hypothesis testing. 
#Null Hypothesis: There is no difference in the means
#Alternative Hypothesis: There is difference in the means
#95% C.I.

library(PASWR2)
z.test(x4,x3,alternative="two.sided", mu=0, sigma.x=sd(x4),sigma.y=sd(x3), conf.level = 0.95)

#p-value is less than 0.05. hence, null hypothesis is rejected.

#KNN MODEL
#Corpus is collection of data on which we can apply text mining in order to device inferences
#Using columns 'Description' as independent variables and 'fraudulent' as dependent variables
#DataPreprocessing on column 'Description'

library(tm)

#Column 'Description' is converted to a DocumentTermMatrix and then converted to a dataframe to which the column 'fraudulent' is added to be treated as outcome.
corpus_object <- Corpus(VectorSource(x1))

#Pre-processing on the dataset
#To remove any punctuations
corpus_object <- tm_map(corpus_object, removePunctuation)

#To remove any stopwords based on en dictionary
corpus_object <- tm_map(corpus_object, removeWords, stopwords(kind = "en"))

#Remove common words in English. Normalization process. Also termed as Portwer's stemming algorithm
corpus_object <- tm_map(corpus_object, stemDocument)

#Converting all the remaining high frequency words to a DocumentTermMatrix
frequencies <- DocumentTermMatrix(corpus_object)

#remove terms that are occuring more sparse than the threshold set as 0.995. We then have only the words occurring with high frequency
sparse_data_desc <- removeSparseTerms(frequencies, 0.995)

#Creating dataframe
sparse_data_jpdesc <- as.data.frame(as.matrix(sparse_data_desc))

#Assigning column names to the dataframe
colnames(sparse_data_jpdesc) <- make.names(colnames(sparse_data_jpdesc))

#Adding column 'fraudulent' to the dataframe to be treated as output
sparse_data_jpdesc$fraudulent <- job_posting$fraudulent

#Removing duplicate column
colnames(sparse_data_jpdesc) <- make.unique(colnames(sparse_data_jpdesc), sep = "_")

set.seed(2000)

#Labels though have numerical values, are nominal. R reads them as numeric variables. Hence labels need to be converted to nominal using factor function.
sparse_data_jpdesc$fraudulent=factor(sparse_data_jpdesc$fraudulent)

#Creating Training and Validation datasets

library(caret)
#predict on a test set of 100 observations
test_index <- 1:100

#Without labels
train.jp <- sparse_data_jpdesc[-test_index,]
test.jp <- sparse_data_jpdesc[test_index,]

#With Labels
train.fr <- sparse_data_jpdesc$fraudulent[-test_index]
test.fr <- sparse_data_jpdesc$fraudulent[test_index]

library(class)
knn.5 <- knn(train.jp,test.jp,train.fr,k=5)

#Calculating Accuracy
library(Metrics)
accuracy(test.fr,knn.5)

#Ans: Accuracy turned out to be 99%
#KNN tried using other columns but got error as : 'There are too many ties for the given knn' This occurs when there are more than expected equidistant points. Tried till knn=1000 more than which the R session was aborted

#LOGISTIC REGRESSION MODEL

#R reads labels as numeric variables. Using function factor to convert it into nominal.
x10f = factor(x10)

#Creating training and Validation datasets
job_posting=job_posting[sample(nrow(job_posting)),]
select.data = sample (1:nrow(job_posting),0.8*nrow(job_posting))

train.data <- job_posting[select.data,]
test.data <- job_posting[-select.data,]

train.label <- train.data$fraudulent
test.label <- test.data$fraudulent

#Logistic Regression Model
fit <- glm(x10f ~ x2+x3+x4+x5+x6+x7+x8+x9 , data=train.data, family=binomial())
summary(fit)

#Using Forward selection
base=glm(x10f~x7, data=train.data, family=binomial())
model1= step(base, scope=list(upper=fit,lower=~1),direction="both", trace=F)
summary(model1)

#Using Backward elimination
model2= step(fit,direction="backward",trace=T)
summary(model2)

#Accuracy Prediction

predict(fit,type="response",newdata=test.data)
prob=predict(fit,type="response",newdata=test.data)

#Cut-off value to calculate accuracy is 0.5
for(i in 1:length(prob)){
  if (prob[i] > 0.5){
    prob[i]=1}
  else {
    prob[i]=0
  }
}

library(Metrics)
accuracy(test.label,prob)

#SVM MODEL

#Selecting columns which will be considered for SVM and creating a separate dataset

job_posting=job_posting[sample(nrow(job_posting)),]
select.data = sample (1:nrow(job_posting),0.8*nrow(job_posting))

train.data <- job_posting[select.data,]
test.data <- job_posting[-select.data,]

train.label <- train.data$fraudulent
test.label <- test.data$fraudulent


#Using Radial Kernel with default parameters

library(e1071)
svm_model <- svm(x10f~x2+x3+x4+x5+x6+x7+x8+x9 , data=train.data, method="C-classification", kernel="radial")
svm_model$cost
svm_model$gamma
pred_train<- predict(svm_model,train.data)
pred_train
mean(pred_train==train.label)
pred_test <-predict(svm_model,test.data)
mean(pred_train==test.label)

#Using Radial Kernel with Best Parameters, Gamma and Cost
tune_out <- tune.svm(x=x2+x3+x4+x5+x6+x7+x8+x9,y=x10f,data=train.data, gamma=10^(-3:3),cost=c(0.01,0.1,1,10,100,1000),kernel="radial")
tune_out
tune_out$best.parameters$cost
tune_out$best.parameters$gamma

svm_model <- svm(x10f~ x2+x3+x4+x5+x6+x7+x8+x9, data=train.data, method="C-classification", kernel="radial",cost=tune_out$best.parameters$cost,gamma=tune_out$best.parameters$gamma)
pred_train<- predict(svm_model,train.data)
pred_train
mean(pred_train==train.label)
pred_test <-predict(svm_model,test.data)
mean(pred_train==test.label)
print(svm_model)

#Using Linear Kernel
svm_model <- svm(x10f~ x2+x3+x4+x5+x6+x7+x8+x9, data=train.data, method="C-classification", kernel="linear")
pred_train<- predict(svm_model,train.data)
pred_train
mean(pred_train==train.label)
pred_test <-predict(svm_model,test.data)
mean(pred_train==test.label)

#Feature Extraction using column 'Description'

library(dplyr)

description <- job_posting$description

#To convert the column to char
description <- as.character(description)

#Feature Extraction on Genuine Postings

#Filter out genuine postings
Genuine_description <- filter(job_posting,fraudulent==0)

#To convert the column to data frame
description_G <- as.data.frame(Genuine_description$description)

#To count the characters in each value of the column 'description'
count <- seq(1, nrow(description_G), 1)
count

#To convert the characters to lowercase
lowerC_description_G <- sapply(count, function(c){tolower(description_G[c, 1])})
head(lowerC_description_G,1)

library(stringr)

#To replace all non-alphanumeric characters.
del_special_chars_description_G <- sapply(count, function(l){str_replace_all(lowerC_description_G[l], "[^[:alnum:]]", " ")})

library(tm)

#Converting into a Corpus object as we did while performing KNN

del_special_chars_description_G <- VCorpus(VectorSource(del_special_chars_description_G))

#Remove stop words
del_special_chars_description_G <- tm_map(del_special_chars_description_G, removeWords, stopwords(kind = "en"))

#Remove punctuation
del_special_chars_description_G <- tm_map(del_special_chars_description_G, removePunctuation)

#Remove WhiteSpace
white_space_cleanup_description_G <- tm_map(del_special_chars_description_G, stripWhitespace)

inspect(white_space_cleanup_description_G[1])

library(SnowballC)

#Stemming to remove English language commoners
stemming_description_G <- tm_map(white_space_cleanup_description_G, stemDocument)
text_description_G <- tm_map(stemming_description_G, stemDocument,language = "english")

#Converting to TermDocumentMatrix

docterm_corpus_description_G <- TermDocumentMatrix(text_description_G)

#Remove sparse terms occurring lesser than the threshold set as 0.99
review_Gen = removeSparseTerms(docterm_corpus_description_G, 0.99)

#Find Freq Terms
findFreqTerms(review_Gen, 1000)

#Converting to a matrix
description_matrix_G <- as.matrix(review_Gen)

v_gen <- sort(rowSums(description_matrix_G),decreasing=TRUE)
d_gen <- data.frame(word = names(v_gen),freq=v_gen)
#head(d, 10)

# Creating a word cloud. The larger the words, higher the frequency.

set.seed(2020)
library(wordcloud)
wordcloud(words = d_gen$word, freq = d_gen$freq, min.freq = 1,
          max.words=150, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


#Filter out fake postings
Fake_description <- filter(job_posting,fraudulent==1)

#To convert the column to data frame
description_F <- as.data.frame(Fake_description$description)

#To count the characters in each value of the column 'description'
count <- seq(1, nrow(description_F), 1)
count

#To convert the characters to lowercase
lowerC_description_F <- sapply(count, function(c){tolower(description_F[c, 1])})
head(lowerC_description_F,1)

library(stringr)

#To replace all non-alphanumeric characters.
del_special_chars_description_F <- sapply(count, function(l){str_replace_all(lowerC_description_F[l], "[^[:alnum:]]", " ")})

library(tm)

#Converting into a Corpus object as we did while performing KNN

del_special_chars_description_F <- VCorpus(VectorSource(del_special_chars_description_F))

#Remove stop words
del_special_chars_description_F <- tm_map(del_special_chars_description_F, removeWords, stopwords(kind = "en"))

#Remove punctuation
del_special_chars_description_F <- tm_map(del_special_chars_description_F, removePunctuation)

#Remove WhiteSpace
white_space_cleanup_description_F <- tm_map(del_special_chars_description_F, stripWhitespace)

inspect(white_space_cleanup_description_F[1])

#Stemming to remove English language commoners
stemming_description_F <- tm_map(white_space_cleanup_description_F, stemDocument)
text_description_F <- tm_map(stemming_description_F, stemDocument,language = "english")

library(SnowballC)

#Converting to TermDocumentMatrix

docterm_corpus_description_F <- TermDocumentMatrix(text_description_F)

#Removing sparse terms above the threshold
review_Fake = removeSparseTerms(docterm_corpus_description_F, 0.99)

#Find high frequency terms
findFreqTerms(review_Fake, 1000)

description_matrix_F <- as.matrix(review_Fake)

v_fake <- sort(rowSums(description_matrix_F),decreasing=TRUE)
d_fake <- data.frame(word = names(v_fake),freq=v_fake)
#head(d, 10)

#Creating wordcloud

set.seed(2020)
wordcloud(words = d_fake$word, freq = d_fake$freq, min.freq = 1,
          max.words=150, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

#To compare the words present in fake and genuine postings

d_gen <- data.frame(lapply(d_gen, as.character), stringsAsFactors=FALSE)

d_fake <- data.frame(lapply(d_fake, as.character), stringsAsFactors=FALSE)

#To identify the rows that exist in d_fake but not in d_gen

d_fake_notin_d <-   d_fake[!d_fake$word %in% d_gen$word, ]

head(d_fake_notin_d)

# Remove NA's
d_fake_notin_d <-   d_fake_notin_d[complete.cases(d_fake_notin_d), ]

head(d_fake_notin_d)

# convert them back to factor and numeric so they can be plotted
d_fake_notin_d$word <- as.factor(d_fake_notin_d$word)
d_fake_notin_d$freq <- as.numeric(d_fake_notin_d$freq)

set.seed(2020)
wordcloud(words = d_fake_notin_d$word, freq = d_fake_notin_d$freq, min.freq = 5,
          max.words=250, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

#GAS OIL AKER SUBSEA CRUI PLANT ULTRA : Present in fake postings but not in general postings

