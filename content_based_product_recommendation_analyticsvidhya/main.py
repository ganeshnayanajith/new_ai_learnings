# Basic Libraries

import numpy as np
import pandas as pd

# Visualization Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Text Handling Libraries

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

df = pd.read_csv('BigBasket Products.csv', index_col='index')
print(df.shape)
print(df.head())

print('Null Data Count In Each Column')
print('-' * 30)
print(df.isnull().sum())

print('-' * 30)
print('Null Data % In Each Column')
print('-' * 30)

# returns the number of rows in the DataFrame
print(f'df.shape[0] : {df.shape[0]}')

print('-' * 30)

for col in df.columns:
    null_count = df[col].isnull().sum()
    total_count = df.shape[0]
    print("{} : {:.2f}".format(col, null_count / total_count * 100))

print('-' * 30)

df = df.dropna()
print(df.shape)

print('-' * 30)

print(df.dtypes)

print('-' * 30)
