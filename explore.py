# Open with VS Code using Python Jupyter extension
# %%
import pandas as pd 

ratings = pd.read_csv('data/ml-20m/ratings.csv')
movies = pd.read_csv('data/ml-20m/movies.csv')
plots = pd.read_csv('data/wikipedia-plots/wiki_movie_plots_deduped.csv')


# %%
print('Total movies rated: ', ratings.movieId.nunique())
print('Total users: ', ratings.userId.nunique())
print('Avg. rating per user: ', ratings.groupby('userId')['rating'].mean().mean())

# %%
ratings.head().T

# %%
movies.head().T

# %%
plots.head().T
