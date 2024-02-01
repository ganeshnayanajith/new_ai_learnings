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

# fig = px.bar(data_frame=counts_df,
#              x='Category',
#              y='Counts',
#              color='Counts',
#              color_continuous_scale='blues',
#              text_auto=True,
#              title=f'Count of Items in Each Category')
#
# fig.write_html("Count of Items in Each Category.html")

counts = df['sub_category'].value_counts()

print(counts)
print('-' * 30)

count_percentage = df['sub_category'].value_counts(1) * 100

print(count_percentage)
print('-' * 30)

counts_df = pd.DataFrame(
    {'sub_category': counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print('unique sub_category values', df['sub_category'].nunique())
print('Top 10 sub_category')
print(counts_df.head(10))
print('Bottom 10 sub_category')
print(counts_df.tail(10))
# fig = px.bar(data_frame=counts_df[:10],
#              x='sub_category',
#              y='Counts',
#              color='Counts',
#              color_continuous_scale='blues',
#              text_auto=True,
#              title=f'Top 10 Bought Sub_Categories')
#
# fig.write_html("Top 10 Bought Sub_Categories.html")
#
# fig = px.bar(data_frame=counts_df[-10:],
#              x='sub_category',
#              y='Counts',
#              color='Counts',
#              color_continuous_scale='blues',
#              text_auto=True,
#              title=f'Bottom 10 Bought Sub_Categories')
#
# fig.write_html("Bottom 10 Bought Sub_Categories.html")

column = 'brand'
counts = df[column].value_counts()

print(counts)
print('-' * 30)

count_percentage = df[column].value_counts(1) * 100

print(count_percentage)
print('-' * 30)

counts_df = pd.DataFrame(
    {column: counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print('unique ' + str(column) + ' values', df[column].nunique())
print('Top 10 ' + str(column))
print(counts_df.head(10))
print(counts_df[counts_df['Counts'] == 1].shape)
print('Bottom 10 ' + str(column))
print(counts_df.tail(10))
# fig = px.bar(data_frame=counts_df.head(10),
#              x=column,
#              y='Counts',
#              color='Counts',
#              color_continuous_scale='blues',
#              text_auto=True,
#              title=f'Top 10 Brand Items based on Item Counts')
#
# fig.write_html("Top 10 Brand Items based on Item Counts.html")


column = 'type'
counts = df[column].value_counts()
count_percentage = df[column].value_counts(1) * 100
counts_df = pd.DataFrame(
    {column: counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print('unique ' + str(column) + ' values', df[column].nunique())
print('Top 10 ' + str(column))
print(counts_df.head(10))
print(counts_df[counts_df['Counts'] == 1].shape)
# fig = px.bar(data_frame=counts_df.head(10),
#              x='type',
#              y='Counts',
#              color='Counts',
#              color_continuous_scale='blues',
#              text_auto=True,
#              title=f'Top 10 Types of Products based on Item Counts')
#
# fig.write_html("Top 10 Types of Products based on Item Counts.html")

print(df['rating'].describe())
print('-' * 30)

df['rating'].hist(bins=10)

# plt.show()

result = pd.cut(df.rating, bins=[0, 1, 2, 3, 4, 5]).reset_index().groupby(['rating'], observed=False).size()

print(result)
print('-' * 30)

df['discount'] = (df['market_price'] - df['sale_price']) * 100 / df['market_price']
print(df['discount'].describe())
print('-' * 30)

result = pd.cut(df.discount, bins=[-1, 0, 10, 20, 30, 40, 50, 60, 80, 90, 100]).reset_index().groupby(
    ['discount'], observed=False).size()

print(result)
print('-' * 30)

df['discount'].hist()

# plt.show()
