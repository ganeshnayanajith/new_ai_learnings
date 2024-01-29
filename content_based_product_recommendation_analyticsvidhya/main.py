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
