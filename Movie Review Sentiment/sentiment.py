#Tutorial - http://pythonforengineers.com/build-a-sentiment-analysis-app-with-movie-reviews/

#Dataset: movie reviews corpus from NLTK

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


# all_words = movie_reviews.words() #all the words in all the reviews
# print(all_words)	

# print(movie_reviews.fileids())	#print list of all files

# freq_dist = nltk.FreqDist(all_words)	#Frequency of all words
# print(freq_dist.most_common(10))	#Most of them are stop words!


##### Now main program #####

#Read https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification for how NaiveBayesClassifier work

def naive_bayes_input(words):
	useful_words = [word for word in words if word not in stopwords.words('english')]
	words_dict = dict([(word, True) for word in useful_words])
	return words_dict	#This is how NaiveBayesClassifier expects input

#The sentiment analysis code is just a machine learning algorithm that has been trained to identify positive/negative reviews.

negative_reviews = []
for file in movie_reviews.fileids('neg'):
	words = movie_reviews.words(file)
	negative_reviews.append((naive_bayes_input(words), 'negative'))

positive_reviews = []
for file in movie_reviews.fileids('pos'):
	words = movie_reviews.words(file)
	positive_reviews.append((naive_bayes_input(words), 'positive'))

# print(len(negative_reviews), len(positive_reviews))

train_set = negative_reviews[:800] + positive_reviews[:800]
test_set = negative_reviews[800:] + positive_reviews[800:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy*100)	#Accuracy of 72.5% on the dataset