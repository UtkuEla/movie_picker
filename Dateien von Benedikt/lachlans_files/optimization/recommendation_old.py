import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

def compute_similarities(df):
    # Bessere Vorverarbeitung der Texte
    df['overview'] = df['overview'].fillna('')
    df['overview'] = df['overview'].apply(preprocess_text)
    
    # Verbesserte TF-IDF für Beschreibungen mit n-grams
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Verwende Unigramme und Bigramme
        max_features=5000,   # Begrenze Features auf die wichtigsten
        min_df=2             # Ignoriere zu seltene Wörter
    )
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim1 = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Verbesserte Metadata-Suppe mit besserer Gewichtung
    # Vorbereiten der Metadaten mit Gewichtung
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
    
    # Kombiniere gewichtete Metadaten
    df['metadata_soup'] = (
        df['genres_weighted'] + ' ' + 
        df['director_weighted'] + ' ' + 
        df['cast_weighted'] + ' ' + 
        df['keywords_weighted'] + ' ' + 
        df['title'].fillna('')
    )
    
    df['metadata_soup'] = df['metadata_soup'].apply(preprocess_text)
    
    # TF-IDF auf Metadaten
    metadata_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2
    )
    metadata_matrix = metadata_vectorizer.fit_transform(df['metadata_soup'])
    cosine_sim2 = linear_kernel(metadata_matrix, metadata_matrix)
    
    # Kombinierte Ähnlichkeit mit optimierten Gewichten
    cosine_sim_combined = 0.1 * cosine_sim1 + 0.9 * cosine_sim2
    
    # Optional: Ähnlichkeit um Popularität und Bewertung ergänzen
    if 'vote_average' in df.columns and 'vote_count' in df.columns:
        sim_bonus = calculate_popularity_similarity(df)
        # Normalisierte Bonus-Ähnlichkeit integrieren mit geringem Gewicht
        cosine_sim_combined = 0.9 * cosine_sim_combined + 0.1 * sim_bonus
    
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

def calculate_popularity_similarity(df):
    """Erstellt eine Ähnlichkeitsmatrix basierend auf Popularität und Bewertung"""
    # Extrahiere Features
    features = df[['vote_average', 'vote_count']].copy()
    
    # Fülle NaN-Werte
    features['vote_average'] = features['vote_average'].fillna(features['vote_average'].mean())
    features['vote_count'] = features['vote_count'].fillna(0)
    
    # Normalisiere Features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Berechne Ähnlichkeit
    similarity = cosine_similarity(scaled_features)
    
    return similarity

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
    
    # Falls genre gegeben, filtern mit verbesserter Genre-Erkennung
    if genre:
        cleaned_genre = genre.lower().strip()
        filtered_sim_scores = []
        
        for i, score in sim_scores:
            if i == idx:  # Überspringe den eigenen Film
                continue
                
            movie_genres = [g.lower().strip() for g in df.iloc[i]['genres'].split(",")]
            
            # Direkte Genre-Übereinstimmung
            if cleaned_genre in movie_genres:
                # Bonus für exaktes Genre
                filtered_sim_scores.append((i, score * 1.1))
            # Fuzzy matching falls kein direkter Treffer
            elif any(cleaned_genre in g for g in movie_genres):
                filtered_sim_scores.append((i, score))
    else:
        filtered_sim_scores = [(i, score) for i, score in sim_scores if i != idx]
    
    # Top N holen, wobei N benutzerdefiniert ist
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    movie_indices = [i[0] for i in filtered_sim_scores]
    
    # Extrahiere Ähnlichkeitsgrade für die Ausgabe
    similarities = [score for _, score in filtered_sim_scores]
    
    # Füge Ähnlichkeitsgrade zu den Ergebnissen hinzu
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity_score'] = similarities
    
    return result_df