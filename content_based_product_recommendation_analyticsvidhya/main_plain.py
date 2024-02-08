import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('BigBasket Products.csv', index_col='index')
print(df.shape)
print(df.head())

print('Null Data Count In Each Column')

print(df.isnull().sum())

print('Null Data % In Each Column')

# returns the number of rows in the DataFrame
print(f'df.shape[0] : {df.shape[0]}')

for col in df.columns:
    null_count = df[col].isnull().sum()
    total_count = df.shape[0]
    print("{} : {:.2f}".format(col, null_count / total_count * 100))

df = df.dropna()
print(df.shape)

print(df.dtypes)

counts = df['category'].value_counts()

print(counts)

count_percentage = df['category'].value_counts(1) * 100

print(df['category'].value_counts(1))

print(count_percentage)

counts_df = pd.DataFrame(
    {'Category': counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print(counts_df)

counts = df['sub_category'].value_counts()

print(counts)

count_percentage = df['sub_category'].value_counts(1) * 100

print(count_percentage)

counts_df = pd.DataFrame(
    {'sub_category': counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print('unique sub_category values', df['sub_category'].nunique())
print('Top 10 sub_category')
print(counts_df.head(10))
print('Bottom 10 sub_category')
print(counts_df.tail(10))

column = 'brand'
counts = df[column].value_counts()

print(counts)

count_percentage = df[column].value_counts(1) * 100

print(count_percentage)

counts_df = pd.DataFrame(
    {column: counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print('unique ' + str(column) + ' values', df[column].nunique())
print('Top 10 ' + str(column))
print(counts_df.head(10))
print(counts_df[counts_df['Counts'] == 1].shape)
print('Bottom 10 ' + str(column))
print(counts_df.tail(10))

column = 'type'
counts = df[column].value_counts()
count_percentage = df[column].value_counts(1) * 100
counts_df = pd.DataFrame(
    {column: counts.index, 'Counts': counts.values, 'Percent': np.round(count_percentage.values, 2)})
print('unique ' + str(column) + ' values', df[column].nunique())
print('Top 10 ' + str(column))
print(counts_df.head(10))
print(counts_df[counts_df['Counts'] == 1].shape)

result = df['rating'].describe()

df['rating'].hist(bins=10)

result = pd.cut(df.rating, bins=[0, 1, 2, 3, 4, 5]).reset_index().groupby(['rating'], observed=False).size()

df['discount'] = (df['market_price'] - df['sale_price']) * 100 / df['market_price']
result = df['discount'].describe()

result = pd.cut(df.discount, bins=[-1, 0, 10, 20, 30, 40, 50, 60, 80, 90, 100]).reset_index().groupby(
    ['discount'], observed=False).size()

df['discount'].hist()

ax = df.plot.scatter(x='rating', y='discount')

df2 = df.copy()
rmv_spc = lambda a: a.strip()
get_list = lambda a: list(map(rmv_spc, re.split('& |, |\*|n', a)))

for col in ['category', 'sub_category', 'type']:
    df2[col] = df2[col].apply(get_list)


def cleaner(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


for col in ['category', 'sub_category', 'type', 'brand']:
    df2[col] = df2[col].apply(cleaner)


def couple(x):
    return ' '.join(x['category']) + ' ' + ' '.join(x['sub_category']) + ' ' + x['brand'] + ' ' + ' '.join(x['type'])


df2['product_classification_features'] = df2.apply(couple, axis=1)

result = df2['product_classification_features'].head()


def recommend_most_popular(col, col_value, top_n=5):
    return df[df[col] == col_value].sort_values(by='rating', ascending=False).head(top_n)[['product', col, 'rating']]


result = recommend_most_popular(col='category', col_value='Beauty & Hygiene')

result = recommend_most_popular(col='sub_category', col_value='Hair Care')

result = recommend_most_popular(col='brand', col_value='Amul')

result = recommend_most_popular(col='type', col_value='Face Care')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['product_classification_features'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim)


def content_recommendation_v1(title):
    a = df2.copy().reset_index().drop('index', axis=1)
    index = a[a['product'] == title].index[0]
    top_n_index = list(cosine_sim_df[index].nlargest(10).index)
    try:
        top_n_index.remove(index)
    except:
        pass
    similar_df = a.iloc[top_n_index][['product']]
    similar_df['cosine_similarity'] = cosine_sim_df[index].iloc[top_n_index]
    return similar_df


title = 'Water Bottle - Orange'
result = content_recommendation_v1(title)

title = 'Dark Chocolate- 55% Rich In Cocoa'
result = content_recommendation_v1(title)

count2 = CountVectorizer(stop_words='english', lowercase=True)
count_matrix2 = count2.fit_transform(df2['product'])
cosine_sim2 = cosine_similarity(count_matrix2, count_matrix2)
cosine_sim_df2 = pd.DataFrame(cosine_sim2)


def content_recommendation_v2(title):
    a = df2.copy().reset_index().drop('index', axis=1)
    index = a[a['product'] == title].index[0]
    similar_basis_metric_1 = cosine_sim_df[cosine_sim_df[index] > 0][index].reset_index().rename(
        columns={index: 'sim_1'})
    similar_basis_metric_2 = cosine_sim_df2[cosine_sim_df2[index] > 0][index].reset_index().rename(
        columns={index: 'sim_2'})
    similar_df = similar_basis_metric_1.merge(similar_basis_metric_2, how='left').merge(a[['product']].reset_index(),
                                                                                        how='left')
    similar_df['sim'] = similar_df[['sim_1', 'sim_2']].fillna(0).mean(axis=1)
    similar_df = similar_df[similar_df['index'] != index].sort_values(by='sim', ascending=False)
    return similar_df[['product', 'sim']].head(10)


title = 'Water Bottle - Orange'
result = content_recommendation_v2(title)

title = 'Dark Chocolate- 55% Rich In Cocoa'
result = content_recommendation_v2(title)

title = 'Nacho Round Chips'
result = content_recommendation_v2(title)

title = 'Chewy Mints - Lemon'
result = content_recommendation_v2(title)

title = 'Veggie - Fingers'
result = content_recommendation_v2(title)