#convert the dataset from files to a python DataFrame
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


df = pd.read_csv('datasets/movie_data.csv')
reviews = df.review.str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(reviews)

# Estas dos líneas se agregan para eliminar las palabras inútiles
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50])

