import pandas as pd
import string
import fasttext
import re
from re import sub
from sklearn.decomposition import PCA
from matplotlib import pyplot
from io import open
import numpy as np

df=pd.read_csv('sim_fire.csv')

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text


#code source is http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

import re
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "i'm": "i am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you will",
  "you'll've": "you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    text = c_re.sub(replace, text.lower())
    return text


def strip_mentions(text):
    entity_prefixes = ['@']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

for i in range(0,len(df)):
    df.at[i,'text']=strip_links(df.at[i,'text'])
    df.at[i,'text']=expandContractions(df.at[i,'text'])
    df.at[i,'text']=strip_mentions(df.at[i,'text'])

RelevantTweets = df[df['Pass']==1]

NonRelevantTweets = df[df['Pass']==0]

df.text.to_csv('modelTweets.txt',header=False,index=False)

modelTweets = fasttext.skipgram('modelTweets.txt', 'model')


def calcPCA(tweet_dictionary,model):
    pca = PCA(n_components=2)
    X =[]
    for line in tweet_dictionary:
        X.append(model[line])
    result = pca.fit_transform(X)
    return result

resultRelevant = calcPCA(RelevantTweets.text,modelTweets)
resultNonRelevant = calcPCA(NonRelevantTweets.text,modelTweets)


RelevantTweets['PCA1'] = np.array(resultRelevant)[:,0].tolist()
RelevantTweets['PCA2'] = np.array(resultRelevant)[:,1].tolist()

NonRelevantTweets['PCA1'] = np.array(resultNonRelevant)[:,0].tolist()
NonRelevantTweets['PCA2'] = np.array(resultNonRelevant)[:,1].tolist()

RelevantTweets.to_csv('Relevant.csv',index=False)

NonRelevantTweets.to_csv('NonRelevant.csv',index=False)

pyplot.scatter(resultRelevant[:, 0], resultRelevant[:, 1], color="blue", label='remain')

pyplot.scatter(resultNonRelevant[:, 0], resultNonRelevant[:, 1], color = "red", label='removed')
pyplot.title('Fire Example Filtering PCA',size = 20)
pyplot.legend(loc='upper right')
pyplot.savefig('fire.pdf',bbox_inces='tight')
#pyplot.show()
