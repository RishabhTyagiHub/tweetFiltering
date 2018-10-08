# coding=utf-8
import pandas as pd
import string
import fasttext
import re
from re import sub
from sklearn.decomposition import PCA
from matplotlib import pyplot
from io import open
import numpy as np

RelevantTweets = open("RelevantTweets.txt","w", encoding='utf-8')
NonRelevantTweets = open("NonRelevantTweets.txt", "w", encoding='utf-8')
NonRelevantTweet_dictionary = {}
RelevantTweet_dictionary = {}

with open('/Users/RishabhTyagi/Downloads/sim_fire.csv') as f:
    i = 0
    j = 0
    counter = 0
    for line in f.readlines():
        counter+=1
        if counter <= 1: continue
        temp = line.strip().split(',')
        if temp[3] == "":
            NonRelevantTweets.write(temp[1])
            NonRelevantTweet_dictionary[i] = temp[1].lower()
            i+=1
        else:
            RelevantTweets.write(temp[1])
            RelevantTweet_dictionary[j] = temp[1].lower()
            j+=1

def preProcess(tweet_dictionary):




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
    #df = df.iloc[1:]
    ##### Remove the retweet (RT) symbol
    for i in range(0,len(tweet_dictionary)):
        tweet_dictionary[i] = tweet_dictionary[i].replace('RT', '')

    ##### Remove erroneous words

    for i in range(0,len(tweet_dictionary)):
        tweet_dictionary[i] = sub(pattern=r"\d", repl=r"", string=tweet_dictionary[i])

    return tweet_dictionary

NonRelevantTweet_dictionary = preProcess(NonRelevantTweet_dictionary)
RelevantTweet_dictionary = preProcess(RelevantTweet_dictionary)



def writeDicttoFile(tweet_dictionary, keyword):
    lines = tweet_dictionary.values()
    with open(keyword+".txt","w") as file:
        for line in lines:
            file.write(line)

writeDicttoFile(NonRelevantTweet_dictionary, "NonRelevantTweet")
writeDicttoFile(RelevantTweet_dictionary,"RelevantTweet")


modelsNonRelevant = fasttext.skipgram('NonRelevantTweet.txt', 'model')
modelsRelevant = fasttext.skipgram('RelevantTweet.txt', 'model')

#print (modelsNonRelevant['more fire era this time jimin and hoseok'])


def calcPCA(tweet_dictionary,model):
    pca = PCA(n_components=2)
    X =[]
    for line in tweet_dictionary.values():
        X.append(model[line])
    result = pca.fit_transform(X)
    return result

resultRelevant = calcPCA(RelevantTweet_dictionary,modelsRelevant)
resultNonRelevant = calcPCA(NonRelevantTweet_dictionary,modelsNonRelevant)


### The below code creates a csv file to map the PCA values to the tweets for both removed and remained tweets seprately
dfResultsRelevant = pd.DataFrame(data = RelevantTweet_dictionary.items(),columns=['TweetIndex','Tweet'])
dfResultsRelevant['PCA1'] = np.array(resultRelevant)[:,0].tolist()
dfResultsRelevant['PCA2'] = np.array(resultRelevant)[:,1].tolist()
dfResultsRelevant.to_csv('Dataframe_Remained.csv', sep=',', encoding='utf-8')
#print "Remainded", len(dfResultsRelevant)

dfResultsNonRelevant = pd.DataFrame(data = NonRelevantTweet_dictionary.items(),columns=['TweetIndex','Tweet'])
dfResultsNonRelevant['PCA1'] =  np.array(resultNonRelevant)[:,0].tolist()
dfResultsNonRelevant['PCA2'] = np.array(resultNonRelevant)[:,1].tolist()
dfResultsNonRelevant.to_csv('Dataframe_Removed.csv', sep=',', encoding='utf-8')
#print "Removed", len(dfResultsNonRelevant)

pyplot.scatter(resultRelevant[:, 0], resultRelevant[:, 1], color="blue")

pyplot.scatter(resultNonRelevant[:, 0], resultNonRelevant[:, 1], color = "red")
pyplot.show()

