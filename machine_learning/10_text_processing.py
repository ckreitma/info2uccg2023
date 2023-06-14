#convert the dataset from files to a python DataFrame
# Se debe correr una sola vez, con los datos bajados de
# http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# y descomprimidos en la carpeta 'datasets'
import pandas as pd
import os
folder = 'datasets/aclImdb'
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for f in ('test', 'train'):    
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir (path) :
            print(f'Procesando {file}...')
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df._append([[txt, labels[l]]],ignore_index=True)
df.columns = ['review', 'sentiment']
df.to_csv('datasets/movie_data.csv',index=False,encoding='utf-8')
df.head()