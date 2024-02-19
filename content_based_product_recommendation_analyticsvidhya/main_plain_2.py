import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('BigBasket Products.csv', index_col='index')

df = df.dropna()

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
