import pandas as pd


ratings = pd.read_csv('data/ml-20m/ratings.csv')
movies = pd.read_csv('data/ml-20m/movies.csv')

# %% movie title add to ratings
ratings['movie_title'] = ratings['movieId'].map(movies.set_index('movieId')['title'])

# %% preprocess popularity feature
ratings['popularity'] = ratings.groupby('movie_title')['movieId'].transform('size')

# %%
ratings.to_csv('data/rec_data.csv', index=False)
