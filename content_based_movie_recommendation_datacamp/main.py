# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
print(metadata.head(3))

# Print plot overviews of the first 5 movies.
print(metadata['overview'].head())

# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
# print(tfidf_matrix)
print(tfidf_matrix.shape)

print(tfidf.get_feature_names_out()[5000:5010])

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#
# print(cosine_sim)
# print(cosine_sim.shape)
#
# print(cosine_sim[1])

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

print(indices)
print(metadata.index)
print(indices[:10])
