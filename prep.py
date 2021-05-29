# %%
import pandas as pd
from similarities import jaccard_similarity_big_data, jaccard_similarity_small_data, jaccard_similarity_score_big_data
from nlp import nlp_prep

def prep_data(ratings, movies, plots):

    ratings = prep_ratings(ratings, movies)
    movie_genres = prep_movie_genres(movies)
    movie_plots = prep_movie_plots(plots)

    ratings.to_csv('data/movie_ratings.csv', index=False)
    movie_genres.to_csv('data/movie_genres.csv')
    movie_plots.to_csv('data/movie_plots.csv')


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



def prep_movie_plots(plots):
    # Generate atable of titles and plots. 
    # Also preprocess the plots with nlp
    
    movie_plots = plots[["Title", "Plot"]]
    movie_plots.columns = ['title', 'plot']
    movie_plots = movie_plots.set_index('title')
    movie_plots["plot_nlp"] = movie_plots["plot"].apply(nlp_prep)

    return movie_plots


ratings = pd.read_csv('data/ml-20m/ratings.csv')
movies = pd.read_csv('data/ml-20m/movies.csv')
plots = pd.read_csv('data/wikipedia-plots/wiki_movie_plots_deduped.csv')
prep_data(ratings, movies, plots)
# %% Play-yard / test functions

# movie_genres = prep_movie_genres(movies)

# # %%
# jaccard_similarity_small_data(movie_genres[:1000])

# # %%
# target_movie = movie_genres.loc['Jumanji (1995)']
# jaccard_similarity_big_data(movie_genres, target_movie)


# movie_plots = prep_movie_plots(plots)