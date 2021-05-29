from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from itertools import combinations
import pandas as pd

def jaccard_similarity_small_data(movie_genres):
    """Calculate Jaccard similarity score for a small data one-hot encoded table.
    Useful to test the multi-processing function for big data that follows. 

    Args:
        movie_genres (pd.DataFrame): One hot-encoded dataframe with rows: movies, binary columns: genres

    Returns:
        pd.DataFrame: Squareform matrix with rows: movies, colums: movies, values: jaccard similarities
    """    

    assert movie_genres.shape[0] <= 1000

    # Calculate all pairwise distances
    jaccard_distances = pdist(movie_genres.values, metric='jaccard')

    # Convert the distances to a square matrix
    jaccard_similarity_array = 1 - squareform(jaccard_distances)

    # Wrap the array in a pandas DataFrame
    jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=movie_genres.index, columns=movie_genres.index)

    # Print the top 5 rows of the DataFrame
    print(jaccard_similarity_df.head())

    return jaccard_similarity_df

# Similar to jaccard_similariy_score from sklearn - 
# A reminder that sklearn's jacccard_similarity score is not appropriate for a RecSys
def jaccard_similarity_score_big_data(movie_genres):
    movie_genres = movie_genres.set_index('title')
    jaccard_similarity_df = 1 - pairwise_distances(movie_genres, metric='hamming')
    jaccard_similarity_df = pd.DataFrame(jaccard_similarity_df, index=movie_genres.index, columns=movie_genres.index)
    
    return jaccard_similarity_df

# Return a series with multi index: movie-rest of the movies and similarity
def jaccard_similarity_big_data(movie_genres, target_movie):
    from sklearn.metrics import jaccard_score
    from functools import partial
    import multiprocessing as mp
    partial_jaccard = partial(jaccard_score, target_movie)
    
    X = movie_genres
    with mp.Pool() as pool:
        results = pool.map(partial_jaccard, [row for row in X.values])

    return pd.DataFrame({'title': X.index, 'jaccard_similarity': results})
    

def calc_cosine_similarity(tfidf_movie_plots):

    assert tfidf_movie_plots.shape[0] <= 1000

    # Calculate the cosine similarity among movie embeddings
    cosine_similarity_array = cosine_similarity(tfidf_movie_plots)
    cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_movie_plots.index, columns=tfidf_movie_plots.index)

    return cosine_similarity_df

def cosine_similarity_user_prof(tfidf_movie_plots, user_prof, movies_enjoyed):

        # Find subset of tfidf_movie_plots that does not include movies in list_of_movies_enjoyed
        tfidf_subset_df = tfidf_movie_plots.drop(movies_enjoyed, axis=0)

        # Calculate the cosine_similarity and wrap it in a DataFrame
        similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
        similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
        
        return similarity_df