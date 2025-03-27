from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re

def compute_similarities(df, cos1=0.4, cos2=0.6):
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
    def process_genres(genres):
        genres_list = genres.split(',')
        main_genres = genres_list[:1]
        return ' '.join(main_genres + genres_list)
        
    df['genres_weighted'] = df['genres'].apply(process_genres)

    df['cast_weighted'] = df['cast'].fillna('').apply(
        lambda x: ' '.join([actor.strip() for actor in x.split(',')[:3]]) 
    )
    
    df['director_weighted'] = df['director'].fillna('').apply(
        lambda x: ' '.join([x.strip() + ' ' + x.strip()])
    )
    
    df['keywords_weighted'] = df['keywords'].fillna('').apply(
        lambda x: ' '.join([kw.strip() for kw in x.split(',')])
    )
    
    # Combine meta data to meta-data-soup
    df['metadata_soup'] = (
        df['genres'] + ' ' + 
        df['director_weighted'] + ' ' + 
        df['cast_weighted'] + ' ' + 
        df['keywords_weighted'] + ' ' + 
        df['title'].fillna('')
    )

    # print(f"Meta Data Soup: {df['metadata_soup'].sample(5)}") 
    
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

    # Scaling before normalisation
    # print(f"cosine_sim1 Min: {cosine_sim1.min()}, Max: {cosine_sim1.max()}")
    # print(f"cosine_sim2 Min: {cosine_sim2.min()}, Max: {cosine_sim2.max()}")

    # normalise matrixes
    scaler = MinMaxScaler()
    cosine_sim1 = scaler.fit_transform(cosine_sim1)
    cosine_sim2 = scaler.fit_transform(cosine_sim2)

    # Scaling after normalisation
    # print(f"cosine_sim1 Min: {cosine_sim1.min()}, Max: {cosine_sim1.max()}")
    # print(f"cosine_sim2 Min: {cosine_sim2.min()}, Max: {cosine_sim2.max()}")
    
    # Combine and weight matrixes
    cosine_sim_combined = cos1 * cosine_sim1 + cos2 * cosine_sim2
    
    # print(f"Matrixwerte Zeilen 1 - 5: {cosine_sim_combined[:5]}")
    
    return cosine_sim_combined


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text