## Sara's Genre Filtering ##

import pandas as pd

""" 
# Function to get all unique genres in the dataset -> this allows to provide a list of available genres before user input
def get_all_genres(movies_df):
    unique_genres = set(genre.strip().title() for genres in movies_df['genres'].dropna() for genre in genres.split(','))
    return sorted(unique_genres)
"""

# Function to filter movies by genre (case-insensitive) -> without this if user wrote 'action' instead of 'Action', no movies would come up 

def filter_movies_by_genre(movies_df, selected_genre):
    cleaned_genre = selected_genre.lower().strip()
    genre_filter = movies_df["genres"].apply(lambda x: cleaned_genre in [g.lower().strip() for g in x.split(",")])
    return movies_df.loc[genre_filter][0:10].round(2)