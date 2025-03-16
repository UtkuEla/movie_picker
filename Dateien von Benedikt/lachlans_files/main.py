## Search Functions ##

from data_utils import load_data
from genre_filter import get_all_genres, filter_movies_by_genre
from recommendation import compute_similarities, get_recommendations_filtered
from tabulate import tabulate

## Search ##
def main():
    
    df = load_data("movies_cleaned_hard.parquet")  ###21## <---- Update this path to your local path #####
    available_genres = get_all_genres(df)
    available_genres_lower = [genre.lower() for genre in available_genres]
    cosine_sim1, cosine_sim2, cosine_sim_combined = compute_similarities(df)
    
    while True:
        print("Welcome to the Movie Recommendation System")
        print("Please choose an option:")
        print("1. Explore by genre (Branch 1)")
        print("2. Search directly by movie title (Branch 2)")
        user_choice = input("Enter 1 or 2: ").strip()
        
        if user_choice == "1":
            print("\nAvailable Genres:\n" + ", ".join(available_genres) + "\n")
            selected_genre = search_genre(df, available_genres_lower, available_genres)
            if selected_genre:
                print(f"\nIf you want to search for more {selected_genre} movies, we can search by movie.")
                search_titles(df, selected_genre, cosine_sim1, cosine_sim2, cosine_sim_combined)
            break
        elif user_choice == "2":
            search_titles(df, None, cosine_sim1, cosine_sim2, cosine_sim_combined)
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

def search_genre(df, available_genres_lower, available_genres, user_genre=None):
    # Exact Match
    if not user_genre:
        user_genre = input("Enter a genre: ").strip().lower()
    if user_genre in available_genres_lower:
        selected_genre = available_genres[available_genres_lower.index(user_genre)]
        filtered_movies = filter_movies_by_genre(df, selected_genre)
        if not filtered_movies.empty:
            print(f"\nTop 10 {selected_genre} Movies (Ranked by Weighted Rating):\n")
            print(tabulate(filtered_movies[['title', 'score', 'vote_average', 'vote_count']], 
                           headers="keys", tablefmt="pretty", showindex=False))
            return selected_genre
        print("\nNo movies found in this genre.")
        return None
    
    # Partial Match
    partial_matches = [genre for genre in available_genres if user_genre in genre.lower()]
    if partial_matches:
        print("\nDid you mean one of these? " + ", ".join(partial_matches))
        user_retry = input("Please re-enter genre: ").strip().lower()
        return search_genre(df, available_genres_lower, available_genres, user_retry)
    
    user_retry = input("\nGenre not found, please try again: ").strip().lower()
    return search_genre(df, available_genres_lower, available_genres, user_retry)

def search_titles(df, genre, cosine_sim1, cosine_sim2, cosine_sim_combined, user_title=None):
    # Exact Match
    if not user_title:
        user_title = input("\nEnter a movie: ").strip().lower()
    if user_title in df['title'].str.lower().values:
        selected_title = df['title'][df['title'].str.lower() == user_title].iloc[0]
        filtered_recommendations = get_recommendations_filtered(df, selected_title, genre, cosine_sim1, cosine_sim2, cosine_sim_combined)
        if isinstance(filtered_recommendations, str):
            print(f"\n{filtered_recommendations}")
            return
        if genre:
            print(f"\nTop 10 {genre} Movies Similar to {selected_title}:\n")
        else:
            print(f"\nTop 10 Movies Similar to {selected_title}:\n")
        print(tabulate(filtered_recommendations[['title', 'score', 'vote_average', 'vote_count']], 
                       headers="keys", tablefmt="pretty", showindex=False))
        return
    
    # Partial Match
    partial_matches = df['title'][df['title'].str.lower().str.contains(user_title, na=False)]
    if not partial_matches.empty:
        print(f"\nDid you mean one of these?\n{'\n'.join(partial_matches[:5])}")
        user_retry = input("Please re-enter title: ").strip().lower()
        return search_titles(df, genre, cosine_sim1, cosine_sim2, cosine_sim_combined, user_retry)
    
    user_retry = input("\nTitle not found, please try again: ").strip().lower()
    return search_titles(df, genre, cosine_sim1, cosine_sim2, cosine_sim_combined, user_retry)

if __name__ == "__main__":
    main()