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

counts = df['category'].value_counts()

print(counts)
print('-' * 30)

count_percentage = df['category'].value_counts(1) * 100

print(df['category'].value_counts(1))
print('-' * 30)

print(count_percentage)
print('-' * 30)

counts_df = pd.DataFrame(
    {'Category': counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print(counts_df)
print('-' * 30)

fig = px.bar(data_frame=counts_df,
             x='Category',
             y='Counts',
             color='Counts',
             color_continuous_scale='blues',
             text_auto=True,
             title=f'Count of Items in Each Category')

fig.write_html("chart.html")
