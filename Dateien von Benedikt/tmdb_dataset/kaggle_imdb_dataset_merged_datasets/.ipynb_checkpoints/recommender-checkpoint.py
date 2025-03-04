
def get_recommendations(title, cosine_sim=cosine_sim, method="series"):
 
    q_movies = pd.read_parquet("../tmbd_exports/quality_movs_weighted_rating.parquet")
    indices = pd.Series(q_movies.index, index = q_movies["title"])
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    
    #Replace NaN with an empty string
    q_movies['overview'] = q_movies['overview']#.fillna('')
    
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(q_movies['overview'])
    
    #Output the shape of tfidf_matrix
    tfidf_matrix.shape
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim.shape

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    if method == "df":
        # Return the top 10 most similar movies
        return q_movies.iloc[movie_indices]

    return q_movies["title"].iloc[movie_indices]
