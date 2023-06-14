#https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958
# Mejor hacer en el shell
# python
# >> import nltk
# >> nltk.download('punkt')
# >> nttk.download('stopwords')
import nltk
from nltk.tokenize import word_tokenize
#function to split text into word
tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
print(tokens)