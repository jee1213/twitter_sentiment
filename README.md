# twitter_sentiment
# Introduction
Some may have heard of a small social experiment where two individuals with different gender exchange their signatures 
at the end of their work email for a week with each other while interacting with their clients, to see if their productivity 
changes based on the apparent gender in the email. The male claimed that when he sent email under female colleague's name, 
clients tend to doubt the professionality of him more, occasionally ask personal questions, resulting in reduced 
productivity. On the other hand, the female counterpart who was using male name, achieved the most productive week 
in her whole career.
We found this interesting, although it is only a report by a single individual and thusn is hard to claim that it is
statistically important. Thus, we wanted to conduct a larger scale experiment using publically available data to see
if the trend is considered real.<br> 
Instead of using work-related emails, we decided to use Twitter data. We collect replies to famous individuals whose background,
including their gender, is known to public. After collecting replies to these known group of people, we perform sentiment analysis
to decide how much insult these individuals get in reply to their tweets. We can then play around with the data to understand
if there is any correlation between the demographics of individuals who are being insulted and other factors.<br> 
For this study, we needed ''famous'' individuals who recieves enough replies. We import the list of 50-most-followed people 
on Twitter from Wikipedia as a starting point.
The reply tweets were collected via twitter streaming API. For modeling insult, which was the hardest part, we collected 
labeled insult from multiple sources: i) Kaggle Imperium challenge, ii) list of Google ban words, and iii) from an academic paper
studying the users who are aggressive / bullying, as well as normal users for comparison. <br>
We perform Natural Language Processing (NLP) and sentiment analysis to those data collected, and do a statistical analysis 
to see if replies to tweets by different people shows statistically meaningful difference in terms of aggression / insult. 
The degree of "aggression/ insult" in a text are modeled based on https://arxiv.org/ftp/arxiv/papers/1604/1604.06648.pdf
https://arxiv.org/pdf/1604.06650.pdf
https://arxiv.org/pdf/1702.06877.pdf
and references therein.

# Code
twitter_NLP_insult_GloVe_RNN.ipynb is the main notebook that can be executed, with all the results.
For collecting large number of tweets from Twitter streaming API, we created a separate notebook
Collect_tweets_50mostpop_users.ipynb. The Twitter credentials are removed as this project became publically available on Github
repository.

# Data
How to use twitter API to collect data and preprocess them:
https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/

## Training data
For training purposes, we started by collecting aggressive users on twitter and their tweets, provided by Despoina Chatzakou (Mean Birds): https://arxiv.org/pdf/1702.06877.pdf
However many of the tweets are unaccessible due to user suspension / authorization issues.
http://www.yichang-cs.com/yahoo/WWW16_Abusivedetection.pdf and dataset suggested therein (e.g. Kaggle challenge) provides insulting comments with verification set.
The list of Google-banned bad words are obtained via https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/
We also apply GloVe and see how the result appears.
## Test data
Using Twitter streaming API, we collect tweets in reply to top 50 most followed users on twitter according to Wikipedia. 
The size of the dataset is 150k. We train the algorithm using training data from three different sources and apply this 
to the test data to classify which of these collected tweets are insulting.  

# Processing tweets
Some words deliver meaning, while some words are there for grammatical reasons. As we want to analyze the sentiment (meaning)
delivered by a sentence, we want to remove those words that don't deliver meanings (i.e. stop words). Also, similar meaning can be 
delivered in various forms, especially in English language: e.g. noun, verb, adjective or adverbs. As a family of word typically
shares the beginning, by 'tokenizing' words (i.e. removing the ending of words), we can reduce the complexity of the text
and focus on the meaning of the words. Processing words thus involves removing stop words, tokenizing words, and depending on 
the application, removing or converting specific features: for example, we remove / convert twitter-specific features 
(e.g. RT, @, url, ...).<br>
Another interesting thing we tried is to convert emojis into its descriptions. Although Python 3.x are able to load unicode
texts automatically (and thus emojis are displayed naturally on Jupyter notebook), they cannot be proceed by NLP while a large 
fraction of tweets include emojis. Thus we implement a emoji to its description using <<unicodedata>> module.
Twitter API returns tweets and its metadata in json format, while emojis are encoded in JAVA unicode escape, as surrogate pairs 
('utf-16', which takes a form of \uxxxx\uxxxx), which is a leagcy format and not widely used anymore. 
Using Regular Expression we detect emojis and replace it by its description (e.g. praying hands)   

# Analysis
We test various sentiment analysis here. We use Vader as a starter, to assess the performance of a typical and easy-to-use 
sentiment analyzer on our training / verification data. We then test various, widely used word embedding and classification 
algorithm to assess the performance of them. Finally we apply the algorithm to the tweets we collected, visualize and 
understand the result.

## Word embedding
In NLP, words need to turn into some numerical representation before fed into an algorithm. The simplest form of this would be 
to encode the corpus one-hot style, where each word has the size of the vocabulary vector attached to it. This creates large and
sparse matrix, which is not ideal in terms of computational resources. Various word embeddings have been proposed to reduce 
the memory usage of the algorithm and computational time, and we try to implement a few.  
### TfidfVectorizer
CountVectorizer (simple token count)-> TfidfTransformer. Probably more suitable for a large corpus with consistent context.
### GloVe
pre-trained unsupervised word clustering / vetorization of words provided by Stanford group. The dimension can be reduced from
order of 10**4 to 10**2 in case of twitter data.

## Classification
### Vader 
Pre-trained positive / negtaive sentiment analyzer. Tweets can be classifies with the intensity of the sentiment. 
### RNN
RNN is a standard choice for sequence modeling (especially text data), where multiple gates are used to connect current
and past words. We implemented RNN with Gate-Recurrent Unit (GRU) using Keras.
