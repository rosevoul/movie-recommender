# %%
import pandas as pd
from recommenders import PopularityRecommender, AvgRankingRecommender, PairRecommender, ContentBasedRecommender
# from evaluate import Evaluator


# Load data
ratings = pd.read_csv('data/rec_data.csv')
movie_genres = pd.read_csv('data/movie_genres.csv', index_col='title')

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
rec_content_based = ContentBasedRecommender(movie_genres)
rec_content_based.recommendations(last_movie_watched)

# %%
# TODO Evaluate

