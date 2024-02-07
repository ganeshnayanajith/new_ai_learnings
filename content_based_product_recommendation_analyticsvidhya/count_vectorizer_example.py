s1 = 'ram is boy'
s2 = 'ram is good'
s3 = 'good is that boy'

from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
import pandas as pd

vectorizer = CountVectorizer(lowercase=True)
X = vectorizer.fit_transform([s1, s2, s3])
X.toarray()
count_vect_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(count_vect_df)

euclidean_distance = distance.cdist(count_vect_df, count_vect_df, metric='euclidean').round(2)
print(euclidean_distance)

manhattan_distance = distance.cdist(count_vect_df, count_vect_df, metric='cityblock').astype('int')
print(manhattan_distance)

jaccard_distance = distance.cdist(count_vect_df, count_vect_df, metric='jaccard').astype('double')
print(jaccard_distance)

cosine_similarity = 1 - distance.cdist(count_vect_df, count_vect_df, metric='cosine').round(2)
print(cosine_similarity)
