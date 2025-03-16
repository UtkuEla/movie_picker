## Benedikt's tf_idf ##

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Schritt 1: Cosinus-Ähnlichkeit aus den Filmbeschreibungen + metadata soup
def compute_similarities(df):
    tfidf = TfidfVectorizer(stop_words='english')
    df['overview'] = df['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim1 = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Metadata soup mit genres, keywords, etc.
    df_soup = df[["genres", "keywords", "title", "cast", "director"]].copy()
    df_soup['soup'] = df_soup.fillna('').astype(str).agg(' '.join, axis=1)
    # metadata_vectorizer = TfidfVectorizer(stop_words='english')
    metadata_matrix = tfidf.fit_transform(df_soup['soup'])
    cosine_sim2 = linear_kernel(metadata_matrix, metadata_matrix)

    # Kombinierte Ähnlichkeit -> 0.8 für overview, 0.2 für soup
    cosine_sim_combined = 0.1 * cosine_sim1 + 0.9 * cosine_sim2
    return cosine_sim1, cosine_sim2, cosine_sim_combined

## Updated for Filtering ##

def get_recommendations_filtered(df, title, genre=None, cosine_sim1=None, cosine_sim2=None, cosine_sim_combined=None, method="combined"):
    # Index vom Film holen
    indices = pd.Series(df.index, index=df['title']).to_dict()
    if title not in indices:
        return "Title not found."
    idx = indices[title]
    
    # Welche Ähnlichkeitsmatrix nutzen
    if method == "cos1":
        cosine_sim = cosine_sim1
    elif method == "cos2":
        cosine_sim = cosine_sim2
    else:
        cosine_sim = cosine_sim_combined
    
    # Ähnlichkeiten berechnen
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Falls genre gegeben, filtern
    if genre:
        cleaned_genre = genre.lower().strip()
        filtered_sim_scores = [
            (i, score) for i, score in sim_scores 
            if cleaned_genre in [g.lower().strip() for g in df.iloc[i]['genres'].split(",")]
        ]
    else:
        filtered_sim_scores = sim_scores
    
    # Top 10 holen, eigenen Film ausschließen
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)
    filtered_sim_scores = [s for s in filtered_sim_scores if s[0] != idx][:10]
    movie_indices = [i[0] for i in filtered_sim_scores]
    
    return df.iloc[movie_indices]