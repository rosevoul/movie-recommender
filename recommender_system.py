# %%
import pandas as pd
from recommenders import *
# from evaluate import Evaluator


# Load data
ratings = pd.read_csv('data/movie_ratings.csv', nrows=1000)
movie_genres = pd.read_csv('data/movie_genres.csv', index_col='title')
movie_plots = pd.read_csv('data/movie_plots.csv', index_col='title', nrows=1000)
tfidf_movie_plots= pd.read_csv('data/movie_plots_tfidf.csv', index_col='title', nrows=1000)
movie_ratings_centered = pd.read_csv('data/movie_ratings_centered.csv', index_col='movie_title', nrows=1000)
user_ratings = pd.read_csv('data/user_ratings.csv', index_col='userId', nrows=1000)
user_ratings_centered = pd.read_csv('data/user_ratings_centered.csv', index_col='userId', nrows=1000)

# %%
# Ignore movies the user has already watched and rated
def filter_watched_movies(training_data, user_id, ignore_watched_movies=False):
    if ignore_watched_movies:
        user_watched_movies = training_data[training_data['userId'] == user_id]['movieId'].unique().tolist()
        filter_watched_movies = (~training_data['movieId'].isin(user_watched_movies))
        training_data = training_data[filter_watched_movies]

    return training_data

# Find the movie last watched by the user
user_last_movies = ratings.sort_values(['userId', 'timestamp'], ascending=True).groupby('userId')['movie_title'].agg('last').reset_index()
user_last_movies.columns = ['userId', 'last_movie_title']

# Showcase with the different Recommender models
user_id = 1
last_movie_watched = user_last_movies.query(f'userId=={user_id}')['last_movie_title'].item()
print('Last Movie Watched: ', last_movie_watched)
training_data = filter_watched_movies(ratings, user_id, ignore_watched_movies=True)
# %%
rec_popularity = PopularityRecommender(ratings)
rec_popularity.recommendations()

# %%
rec_avg_ranking = AvgRankingRecommender(ratings)
rec_avg_ranking.recommendations()


# %%
rec_pair = PairRecommender(ratings)
rec_pair.recommendations(last_movie_watched)


# %%
rec_genre_based = GenreBasedRecommender(movie_genres)
rec_genre_based.recommendations(last_movie_watched)


# %%
rec_plot_based = PlotBasedRecommender(tfidf_movie_plots)
rec_plot_based.recommendations(last_movie_watched='Jack and the Beanstalk')

# %% 
# User profile
movies_enjoyed_list = ['Kansas Saloon Smashers', 'Love by the Light of the Moon',
       'The Martyred Presidents', 'Terrible Teddy, the Grizzly King']

rec_user_prof = UserProfileRecommender(tfidf_movie_plots)
rec_user_prof.recommendations(movies_enjoyed_list)


# %%
rec_col_f = CollaborativeFilteringRecommender(movie_ratings_centered)
rec_col_f.recommendations(last_movie_watched)

# %%
knn_rec = KnnRecommender(user_ratings, user_ratings_centered)
knn_rec.recommendations(user=1)


# %%
svd_rec = SVDRecomender(user_ratings)
svd_rec.recommendations(1)

# # %%
# # TODO Evaluate

