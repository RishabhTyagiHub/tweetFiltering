import gensim
#from gensim import
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from matplotlib import pyplot
sentences = []
sentences1 = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

with open('/Users/RishabhTyagi/Downloads/temp1', 'r') as f:
    lines = f.readlines()
    for line in lines:
        sentences.append(word_tokenize(line))

#print sentences[0]
model = gensim.models.Word2Vec(sentences, sg=1)
#print list(model.wv.vocab)
#print(model['partners'])
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
#\print words
#print result
for i, word in enumerate(words):
    print word, i
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
	#print (word, result[i, 0], result[i, 1])
pyplot.show()
print "Hi"
