#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sun Jul  3 13:07:55 2022
caaturday
❀◕ ‿ ◕❀
"""

# In[0]

'''cosine similarity'''
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

### print(count_matrix.toarray())

similarity_scores = cosine_similarity(count_matrix)

# In[1]
import pandas as pd
import numpy as np

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

# read csv file
df = pd.read_csv(r'/Users/caaturday/Documents/Projects/Machine Learning/movie_dataset.csv')

features = ['keywords', 'cast', 'genres', 'director']

# getting rid of NaNs 
for feature in features:
    df[feature] = df[feature].fillna('')

# creating column in df which combines all selected features

def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("Error: ", row)

df["combined_features"] = df.apply(combine_features, axis=1)

# In[2]

# create matrix from the list of combined features list 
cv = CountVectorizer()

count_matrix = cv.fit_transform(df['combined_features'])


# computing cosine similarity based on the count matrix 
cosine_sim = cosine_similarity(count_matrix)

# In[3]
movie_user_likes = "Avatar"

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse=True)

# printing first 50 movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i+=1
    if i>50:
        break


# In[4]



