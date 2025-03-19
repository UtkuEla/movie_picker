## Search Functions ##

from data_utils import load_data
from interface.genre_filter import get_all_genres, filter_movies_by_genre
from interface.recommendation import compute_similarities, get_recommendations_filtered
from tabulate import tabulate

## Search ##
def main():
    
    df = load_data("movies_cleaned_hard.parquet")
    available_genres = get_all_genres(df)
    available_genres_lower = [genre.lower() for genre in available_genres]
    cosine_sim_combined = compute_similarities(df)
    
    while True:
        print("\n" + "="*50)
        print("Welcome to the Movie Recommendation System")
        print("Please choose an option:")
        print("1. Explore by genre (Branch 1)")
        print("2. Search directly by movie title (Branch 2)")
        print("3. Exit")
        user_choice = input("Enter 1, 2, or 3: ").strip()
        
        if user_choice == "1":
            print("\nAvailable Genres:\n" + ", ".join(available_genres) + "\n")
            selected_genre = search_genre(df, available_genres_lower, available_genres)
            if selected_genre:
                print(f"\nIf you want to search for more {selected_genre} movies, we can search by movie.")
                search_titles(df, selected_genre, cosine_sim_combined)
            
            if ask_restart():
                continue
            else:
                break
                
        elif user_choice == "2":
            search_titles(df, None, cosine_sim_combined)
            
            if ask_restart():
                continue
            else:
                break
                
        elif user_choice == "3":
            print("Thank you for using the Movie Recommendation System. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def ask_restart():
    """Fragt den Benutzer, ob das Programm neu gestartet werden soll."""
    restart = input("\nWould you like to make another search? (y/n): ").strip().lower()
    return restart == 'y' or restart == 'yes'

def get_result_count():
    """Fragt den Benutzer nach der gewünschten Anzahl von Ergebnissen."""
    while True:
        try:
            count = input("\nHow many recommendations would you like to see? (5-50): ").strip()
            count = int(count)
            if 5 <= count <= 50:
                return count
            else:
                print("Please enter a number between 5 and 50.")
        except ValueError:
            print("Please enter a valid number.")

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

def search_titles(df, genre, cosine_sim_combined, user_title=None):
    # Exact Match
    if not user_title:
        user_title = input("\nEnter a movie: ").strip().lower()
    
    # Frage nach der gewünschten Anzahl von Ergebnissen
    result_count = get_result_count()
    
    if user_title in df['title'].str.lower().values:
        selected_title = df['title'][df['title'].str.lower() == user_title].iloc[0]
        filtered_recommendations = get_recommendations_filtered(df, selected_title, genre, cosine_sim_combined, top_n=result_count)
        if isinstance(filtered_recommendations, str):
            print(f"\n{filtered_recommendations}")
            return
        if genre:
            print(f"\nTop {result_count} {genre} Movies Similar to {selected_title}:\n")
        else:
            print(f"\nTop {result_count} Movies Similar to {selected_title}:\n")
        
        # Zeige zusätzlich die Ähnlichkeitswerte an, falls vorhanden
        display_columns = ['title', 'score', 'vote_average', 'vote_count']
        if 'similarity_score' in filtered_recommendations.columns:
            display_columns.append('similarity_score')
            # Formatiere Ähnlichkeitswerte auf 2 Dezimalstellen
            filtered_recommendations['similarity_score'] = filtered_recommendations['similarity_score'].apply(lambda x: round(x, 2))
        
        print(tabulate(filtered_recommendations[display_columns], 
                       headers="keys", tablefmt="pretty", showindex=False))
        return
    
    # Partial Match
    partial_matches = df['title'][df['title'].str.lower().str.contains(user_title, na=False)]
    if not partial_matches.empty:
        print(f"\nDid you mean one of these?\n{''.join(partial_matches[:5])}")
        user_retry = input("Please re-enter title: ").strip().lower()
        return search_titles(df, genre, cosine_sim_combined, user_retry)
    
    user_retry = input("\nTitle not found, please try again: ").strip().lower()
    return search_titles(df, genre, cosine_sim_combined, user_retry)

if __name__ == "__main__":
    main()