import streamlit as st
import pandas as pd
from genre_filter1 import get_all_genres, filter_movies_by_genre
from recommendation1 import compute_similarities, get_recommendations_filtered

st.set_page_config(page_title="Movie Picker", layout="wide")
st.title(":clapper: Welcome to Movie Picker :clapper:")
st.subheader(":rainbow[Passionate Data Cineasts help you to pick your new favorite movie!] :sunglasses:", divider="gray")

@st.cache_data
def load_data():
    df = pd.read_parquet("movies_cleaned_hard.parquet")
    df["title_cleaned"] = df["title"].str.strip().str.lower()
    return df

df = None
cosine_sim1, cosine_sim2, cosine_sim_combined = None, None, None

if "movie_offset" not in st.session_state:
    st.session_state.movie_offset = 0
if "mode" not in st.session_state:
    st.session_state.mode = None

def main():
    global df,  cosine_sim_combined

    if df is None:
        df = load_data()
        cosine_sim_combined = compute_similarities(df)

    if cosine_sim_combined is None:
        st.error("Error loading similarity matrices. Please check data processing.")
        return

    title_to_index = pd.Series(df.index, index=df["title_cleaned"]).to_dict()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Discover by Genre", use_container_width=True):
            st.session_state.mode = "By Genre"
    with col2:
        if st.button("🔎 Find by Movie Name", use_container_width=True):
            st.session_state.mode = "By Movie"
    
    st.markdown("---")
    
    if st.session_state.mode == "By Genre":
        genres = get_all_genres(df)
        selected_genre = st.selectbox("Choose a Genre", ["-- Please select --"] + genres)

        if selected_genre != "-- Please select --":
            movies = filter_movies_by_genre(df, selected_genre)
            total_movies = len(movies)
            movies_to_show = movies.iloc[st.session_state.movie_offset:st.session_state.movie_offset + 10]

            st.write(f"### Movies in {selected_genre} ({st.session_state.movie_offset + 1}-{st.session_state.movie_offset + len(movies_to_show)} of {total_movies})")

            for _, row in movies_to_show.iterrows():
                st.subheader(row['title'])
                st.text(f"Rating: {row['vote_average']} ⭐")
                st.text(f"Director: {row['director']}")
                st.text(f"Actors: {row['cast']}")
                st.write(row['overview'])
                st.markdown("---")

    elif st.session_state.mode == "By Movie":
        movie_input = st.text_input("Enter a Movie Title:")

        if movie_input:
            movie_input_cleaned = movie_input.strip().lower()

            if movie_input_cleaned not in title_to_index:
                st.error("Movie not found in database. Try another title.")
            else:
                recommendations = get_recommendations_filtered(df, df.loc[title_to_index[movie_input_cleaned], "title"], cosine_sim=cosine_sim_combined, method="combined", top_n=10)

                if recommendations is None or isinstance(recommendations, str) or recommendations.empty:
                    st.error("No recommendations found. Please check your input.")
                else:
                    st.write(f"### Movies similar to {movie_input}")
                    for _, row in recommendations.iterrows():
                        st.subheader(row['title'])
                        st.text(f"Rating: {row['vote_average']} ⭐")
                        st.text(f"Director: {row['director']}")
                        st.text(f"Actors: {row['cast']}")
                        st.write(row['overview'])
                        st.markdown("---")

if __name__ == "__main__":
    main()
