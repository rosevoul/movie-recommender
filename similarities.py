from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
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


def calc_cosine_similarity(movie_plots):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Get the TF-IDF transformation of the data
    vectorizer = TfidfVectorizer()
    movie_plots = movie_plots[:1000] 
    vec_data = vectorizer.fit_transform(movie_plots["plot_nlp"])

    tfidf_df = pd.DataFrame(vec_data.toarray(), columns=vectorizer.get_feature_names())
    tfidf_df.index = movie_plots.index
    
    # Calculate the cosine similarity among movie embeddings
    cosine_similarity_array = cosine_similarity(tfidf_df)
    cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_df.index, columns=tfidf_df.index)

    return cosine_similarity_df