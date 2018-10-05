import pandas as pd
import numpy as np
import json
import io
import datetime as dt
import string
import unicodedata
import fasttext
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
import re
from re import sub
from bs4 import BeautifulSoup
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from matplotlib import pyplot

df=pd.read_csv('/Users/RishabhTyagi/Downloads/sim_cancer.csv')
print('Number of observations are: '+str(len(df)))

df=df.tweet.dropna()
df = df.reset_index(drop=True)
print('Number of observations are: '+str(len(df)))

tweet_dictionary = {}
i = 0
for line in df:
        tweet_dictionary[i] = line.lower()
        i += 1

##### Removes all the links in the tweet
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=strip_links(tweet_dictionary[i])

#### REMOVES ALL THE '@'  AND '#' HANDLE MENTIONS IN THE TWEET - HANDLE AND HASHTAGS
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip().to_bytes(4, 'little')
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i]=strip_links(tweet_dictionary[i])

#remove the header from the df
df = df.iloc[1:]
##### Remove the retweet (RT) symbol
for i in range(0,len(df)):
    tweet_dictionary[i] = tweet_dictionary[i].replace('RT', '')

##### Remove erroneous words

for i in range(0,len(tweet_dictionary)):
    tweet_dictionary[i] = sub(pattern=r"\d", repl=r"", string=tweet_dictionary[i])



lines = tweet_dictionary.values()
tokenizedTweetsList = []



with open("data.txt",'w') as file:

    for line in lines:
        file.write(line)

model = fasttext.skipgram('data.txt', 'model')

print model.words
X =  model[model.words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.words)
for i, word in enumerate(words):
    print word, i
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
