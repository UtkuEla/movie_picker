import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

def compute_similarities(df):
    # TF-IDF für Beschreibungen beibehalten (da hier Häufigkeit wichtig ist)
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
    
    # Combine meta data
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
    cosine_sim_combined = 0.6 * cosine_sim1 + 0.4 * cosine_sim2
    
    return cosine_sim_combined

def preprocess_text(text):
    """Text vorverarbeiten: Kleinschreibung, Entfernen von Sonderzeichen usw."""
    if not isinstance(text, str):
        return ""
    # Zu Kleinbuchstaben
    text = text.lower()
    # Entferne Sonderzeichen und behalte nur Buchstaben, Zahlen und Leerzeichen
    text = re.sub(r'[^\w\s]', '', text)
    # Entferne doppelte Leerzeichen
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_recommendations_filtered(df, title, genre=None, cosine_sim=None, method=None, top_n=10):
    """
    Verbesserte Empfehlungsfunktion mit benutzerdefinierbarer Ergebnisanzahl (top_n)
    """
    # Index vom Film holen
    indices = pd.Series(df.index, index=df['title']).to_dict()
    if title not in indices:
        return "Title not found."
    idx = indices[title]
    
    # Ähnlichkeiten berechnen
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    filtered_sim_scores = [(i, score) for i, score in sim_scores if i != idx]
    
    # Top N holen, wobei N benutzerdefiniert ist
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    movie_indices = [i[0] for i in filtered_sim_scores]
    
    # Extrahiere Ähnlichkeitsgrade für die Ausgabe
    similarities = [score for _, score in filtered_sim_scores]
    
    # Füge Ähnlichkeitsgrade zu den Ergebnissen hinzu
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity_score'] = similarities
    
    return result_df.round(2)