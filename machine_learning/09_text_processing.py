import nltk
from nltk.tokenize import word_tokenize

#NLTK provides several stemmer interfaces like Porter stemmer, #Lancaster Stemmer, Snowball Stemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

porter = PorterStemmer()
stems = []

#function to split text into word
tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
for t in tokens:
    stems.append(porter.stem(t))
print(f'Tokens: {tokens}')
print(f'Steam: {stems}')

# Español
palabras = word_tokenize("Estos son los primeros días de muchísimo frío en Paraguay")
snowball_stemmer = SnowballStemmer('spanish')
stemmers2 = [snowball_stemmer.stem(word) for word in palabras]
final2 = [stem for stem in stemmers2 if stem.isalpha() and len(stem) > 1]
print(f'Palabras: {palabras}')
print(f'Final2: {final2}')
