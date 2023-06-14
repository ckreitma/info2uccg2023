#convert the dataset from files to a python DataFrame
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics  import accuracy_score


df = pd.read_csv('datasets/movie_data.csv')

# Dividir el conjunto en test y train
print(f'Dividiendo el conjunto en test y train...{df.shape}')

X_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

print(f'Transformando en vectores...')
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(f'Shape de los vectores:{train_vectors.shape}, {test_vectors.shape}')

print(f'Comenzando aprendizaje...')
clf = MultinomialNB().fit(train_vectors, y_train)

predicted = clf.predict(test_vectors)
print(f'Exactitud Multinomial: {accuracy_score(y_test,predicted)}')

print(f'Comenzando aprendizaje Bernoulli')
clf = BernoulliNB().fit(train_vectors, y_train)

predicted = clf.predict(test_vectors)
print(f'Exactitud Bernoulli: {accuracy_score(y_test,predicted)}')