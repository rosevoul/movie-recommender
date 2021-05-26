# Open with VS Code using Python Jupyter extension
# %%
import pandas as pd 

ratings = pd.read_csv('data/ml-20m/ratings.csv')
movies = pd.read_csv('data/ml-20m/movies.csv')

# %%
print('Total movies rated: ', ratings.movieId.nunique())
print('Total users: ', ratings.userId.nunique())
print('Avg. rating per user: ', ratings.groupby('userId')['rating'].mean().mean())

# %%
ratings.head().T

# %%
movies.head().T
