import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
import random

def compute_similarities(df):
    # TF-IDF for movie-plot (overview)
    df['overview'] = df['overview'].fillna('')
    df['overview'] = df['overview'].apply(preprocess_text)
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2
    )
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim1 = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Meta Soup with weighted entries
    df['genres_weighted'] = df['genres'].fillna('').apply(
        lambda x: ' '.join([genre.strip() + ' ' + genre.strip() for genre in x.split(',')])
    )
    
    df['cast_weighted'] = df['cast'].fillna('').apply(
        lambda x: ' '.join([actor.strip() for actor in x.split(',')[:3]])  # Fokus auf Hauptdarsteller
    )
    
    df['director_weighted'] = df['director'].fillna('').apply(
        lambda x: ' '.join([x.strip() + ' ' + x.strip()])  # Direktor höher gewichten
    )
    
    df['keywords_weighted'] = df['keywords'].fillna('').apply(
        lambda x: ' '.join([kw.strip() for kw in x.split(',')])
    )
    
    # Combine meta data to meta-data-soup
    df['metadata_soup'] = (
        df['genres_weighted'] + ' ' + 
        df['director_weighted'] + ' ' + 
        df['cast_weighted'] + ' ' + 
        df['keywords_weighted'] + ' ' + 
        df['title'].fillna('')
    )
    
    df['metadata_soup'] = df['metadata_soup'].apply(preprocess_text)
    
    # CountVectorizer is used as an alternative to TDF. 
    # https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system?scriptVersionId=161380022&cellId=58
    # We don't want to down-weight the presence of an actor/director if he or she has acted or directed in a lot of movies.
    count_vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2
    )
    count_matrix = count_vectorizer.fit_transform(df['metadata_soup'])
    cosine_sim2 = cosine_similarity(count_matrix)
    
    # Combine and weight matrixes
    cosine_sim_combined = 0.4 * cosine_sim1 + 0.6 * cosine_sim2
    
    return cosine_sim_combined

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_recommendations_filtered(df, title, runs, genre=None, cosine_sim=None, method=None, top_n=10):

    # Index shift: pushes the start index of selected movies on each cycle
    index_shift = runs * 10
    # get movie index
    indices = pd.Series(df.index, index=df['title']).to_dict()
    if title not in indices:
        return "Title not found."
    idx = indices[title]

    # get similarity values 
    sim_scores = list(enumerate(cosine_sim[idx])) 

    filtered_sim_scores = [(i, score) for i, score in sim_scores if i != idx]
        
    # Get the Top-N, whilst Top-N is defined by user 
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)
    # Index shift pushes the start and end value of index by 10 * runs 
    # Does not handle end of list
    filtered_sim_scores = filtered_sim_scores[index_shift:index_shift + top_n] if index_shift < len(filtered_sim_scores) else []

    movie_indices = [i[0] for i in filtered_sim_scores]
    
    # extract similarity for output
    similarities = [score for _, score in filtered_sim_scores]
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity_score'] = similarities
    
    return result_df.round(2)
