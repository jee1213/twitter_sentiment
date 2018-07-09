# twitter_sentiment
# Introduction
The goal of this project is to use twitter streaming API to collect text data, and perform Natural Language Processing (NLP) for sentiment analysis, and do a statistical analysis to see if a tweet in reply to different gender / affiliation, to see if the result shows statistically meaningful difference in terms of aggression / insult. The degree of "aggression/ insult" in a text are modeled based on https://arxiv.org/ftp/arxiv/papers/1604/1604.06648.pdf
https://arxiv.org/pdf/1604.06650.pdf
https://arxiv.org/pdf/1702.06877.pdf
and references therein.
As a pilot survey, we only include 50 significant figures on twitter according to wikipedia (whose gender is known). Accounts for groups / organizations are hand-picked and removed. We only collect tweets that have replies. 

# Code
twitter_NLP_insult_GloVe_RNN.ipynb is the main notebook that can be executed, with all the results.
For collecting large number of tweets from Twitter streaming API, we created a separate notebook
Collect_tweets_50mostpop_users.ipynb

# Data
Uses of twitter APIata collecting and preprocessing step reference:
https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/

## Training data
For training purposes, we started by collecting aggressive users on twitter and their tweets, provided by Despoina Chatzakou (Mean Birds): https://arxiv.org/pdf/1702.06877.pdf
However many of the tweets are unaccessible due to user suspension / authorization issues.
http://www.yichang-cs.com/yahoo/WWW16_Abusivedetection.pdf and dataset provided therein (e.g. Kaggle challenge) provides insulting comments with verification set.
The list of Google-banned bad words are obtained via https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/
We also apply GloVe and see how the result appears.
## Test data
Using Twitter streaming API, we collect tweets in reply to top 50 most followed users on twitter according to Wikipedia. The size of the dataset is 10k tweets to start with. 

# Processing tweets
We perform standard pre-processing of the tweeter text data, which involves:
tokenize, removing stop words, twitter-specific features (e.g. RT, @, ...).

# Analysis
We test various sentiment analysis here. We use Vader as a starter, to assess the performance of a typical and easy-to-use sentiment analyzer on our training / verification data. We then test various widely used word embedding and algorithm to assess the performance of themover Vader. Finally we apply the algorithm to collected tweets, visualize and understand the result.

## Word embedding
### TfidfVectorizer
CountVectorizer (simple token count)-> TfidfTransformer. Probably more suitable for a large corpus with consistent context.
### GloVe
pre-trained unsupervised word clustering / vetorization of words provided by Stanford group.

## Classification
### Vader 
Pre-trained positive / negtaive sentiment analyzer. Tweets can be classifies with the intensity of the sentiment. 
### RNN
RNN is a standard choice for sequence modeling (especially text data). We implemented RNN using Keras.
