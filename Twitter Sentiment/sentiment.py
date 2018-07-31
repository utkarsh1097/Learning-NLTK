import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# print(twitter_samples.fileids())

def remove_smileys(string):
	return string.replace(":", "").replace(")", "").replace("(", "")

def naive_bayes_input(words):
	useful_words = [word for word in words if word not in stopwords.words('english')]
	words_dict = dict([(word, True) for word in useful_words])
	return words_dict	#This is how NaiveBayesClassifier expects input

negative_tweets = []
for string in twitter_samples.strings('negative_tweets.json'):
	# string = remove_smileys(string)
	words = word_tokenize(string)
	negative_tweets.append((naive_bayes_input(words), 'negative'))

positive_tweets = []
for string in twitter_samples.strings('positive_tweets.json'):
	# string = remove_smileys(string)
	words = word_tokenize(string)
	positive_tweets.append((naive_bayes_input(words), 'positive'))

#print(len(negative_tweets), len(positive_tweets))

train_set = negative_tweets[:4000] + positive_tweets[:4000]
test_set = negative_tweets[4000:] + positive_tweets[4000:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy*100)		#97.35%!! This is because tweets are using smileys, which the classifier is learning. :( is learnt as negative and :) as positive.

#Uncomment the line in both for loops, accuracy comes down to ~76%. 