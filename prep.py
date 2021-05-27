# %%
import pandas as pd
from similarities import jaccard_similarity_big_data, jaccard_similarity_small_data, jaccard_similarity_score_big_data


ratings = pd.read_csv('data/ml-20m/ratings.csv')
movies = pd.read_csv('data/ml-20m/movies.csv')


def prep_data(ratings, movies):

    ratings = prep_ratings(ratings, movies)
    movie_genres = prep_movie_genres(movies)

    ratings.to_csv('data/rec_data.csv', index=False)
    movie_genres.to_csv('data/movie_genres.csv')




def prep_ratings(ratings, movies):
    # movie title add to ratings
    ratings['movie_title'] = ratings['movieId'].map(movies.set_index('movieId')['title'])

    # preprocess popularity feature
    ratings['popularity'] = ratings.groupby('movie_title')['movieId'].transform('size')

    return ratings


def prep_movie_genres(movies):
    # Format the data for content-based recommendations
    # Genres: one-hot encoding per movie    

    movie_genres = movies.join(movies.genres.str.get_dummies().astype(bool))
    movie_genres.drop(columns = ['movieId', 'genres'], inplace=True)
    movie_genres = movie_genres.set_index('title')
    
    return movie_genres



# %% Play-yard / test functions

movie_genres = prep_movie_genres(movies)

# %%
jaccard_similarity_small_data(movie_genres[:1000])

# %%
target_movie = movie_genres.loc['Jumanji (1995)']
jaccard_similarity_big_data(movie_genres, target_movie)
