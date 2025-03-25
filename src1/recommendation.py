import pandas as pd

# Globals für Index Shift
index_shift = 0
last_title = ""

def get_recommendations_filtered(df, title, selected_genre=None, cosine_sim_combined=None, top_n=10, counter = False):

    # initialize globals
    global index_shift, last_title

    if title == last_title:
        index_shift += 10
    else:
        index_shift = 0
        last_title = title

    # print(f"Number of cycles: {index_shift}")
    # print(f"Title: {last_title}")
    # print(f"Last title: {last_title}")

    # get movie index
    indices = pd.Series(df.index, index=df['title']).to_dict()
    if title not in indices:
        return "Title not found."
    idx = indices[title]

    # get similarity values 
    sim_scores = list(enumerate(cosine_sim_combined[idx])) 

    filtered_sim_scores = [(i, score) for i, score in sim_scores if i != idx]

    # If genre filter was set by user
    if selected_genre:
        filtered_sim_scores = [
            (i, score) for i, score in filtered_sim_scores if selected_genre in df.iloc[i]['genres']
        ]
        
    # Get the Top-N, whilst Top-N is defined by user 
    filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)
    # Index shift pushes the start and end value of index by 10 * runs 
    # Does not handle end of list
    filtered_sim_scores = filtered_sim_scores[index_shift:index_shift + top_n] if index_shift >= 0 and index_shift < len(filtered_sim_scores) else []

    movie_indices = [i[0] for i in filtered_sim_scores]
    
    # extract similarity for output
    similarities = [score for _, score in filtered_sim_scores]
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity_score'] = similarities

    # result_df = result_df.sort_values(by='score', ascending=False)
    
    return result_df.round(2)
