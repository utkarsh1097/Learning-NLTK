import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

sentence = "The quick brown fox, jumps over the lazy little dog. Hello world!"

words = word_tokenize(sentence)

print(nltk.pos_tag(words))

syn = wordnet.synsets('hello')
print(syn[0].definition())

print(stopwords.words('english'))	#words that carry little or no meaning, but are really common. we would like to get rid of them

para = "The program was open to all women between the ages of 17 and 35, in good health, who had graduated from an accredited high school. "
words = word_tokenize(para)

print(words)

words = [word for word in words if word not in stopwords.words('english')]

print(words)