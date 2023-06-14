#convert the dataset from files to a python DataFrame
import pandas as pd
df = pd.read_csv('datasets/movie_data.csv')
print(df.head())