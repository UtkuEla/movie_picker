## ## Compute similarity matrix once for further use

import pandas as pd
import numpy as np
import json

# Import compute_similarities for similarity matrix 
from matrix_generator import compute_similarities

# Import Parq file and Compose data Frame for matrix

# Params: dataframe, strength-value of cosinus 1 (movie plot), strength of cosinus 2 (meta data soup)
df = pd.read_parquet("movies_cleaned_hard.parquet")
cosine_sim_combined = compute_similarities(df, 0.4, 0.6)
# Save matrix for use in main
np.save("cosine_sim_combined.parquet", cosine_sim_combined)


### Save genres for use in main

# Function to get all unique genres in the dataset -> this allows to provide a list of available genres before user input
def get_all_genres(movies_df):
    unique_genres = set(genre.strip().title() for genres in movies_df['genres'].dropna() for genre in genres.split(','))
    return sorted(unique_genres)

available_genres = get_all_genres(df)

with open("unique_genres.json", "w") as f:
    json.dump(available_genres, f)