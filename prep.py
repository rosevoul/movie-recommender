# %%
import pandas as pd
from similarities import jaccard_similarity_big_data, jaccard_similarity_small_data, jaccard_similarity_score_big_data
from nlp import nlp_prep

def user_prof_tfidf(tfidf_movie_plots, movies_enjoyed):
    # Create a subset of only the movies the user has enjoyed
    movies_enjoyed_df = tfidf_movie_plots.reindex(movies_enjoyed)

    # Generate the user profile by finding the average scores of movies they enjoyed
    user_prof = movies_enjoyed_df.mean()

    return user_prof

def prep_data(ratings, movies, plots):

    movie_ratings = prep_ratings(ratings, movies)
    movie_genres = prep_movie_genres(movies)
    movie_plots = prep_movie_plots(plots)
    tfidf_movie_plots = tfidf_transform(movie_plots)
    movie_ratings_centered = prep_movie_ratings_centered(prep_user_ratings_centered(movie_ratings))
    user_ratings = prep_user_ratings(movie_ratings)
    user_ratings_centered = prep_user_ratings_centered(movie_ratings)

    movie_ratings.to_csv('data/movie_ratings.csv', index=False)
    movie_genres.to_csv('data/movie_genres.csv')
    movie_plots.to_csv('data/movie_plots.csv')
    tfidf_movie_plots.to_csv('data/movie_plots_tfidf.csv')
    movie_ratings_centered.to_csv('data/movie_ratings_centered.csv')
    user_ratings.to_csv('data/user_ratings.csv')
    user_ratings_centered.to_csv('data/user_ratings_centered.csv')



def fill_nan_ratings(user_ratings):
    # Get the average rating for each user 
    avg_ratings = user_ratings.mean(axis=1)
    # Center each users ratings around 0
    user_ratings_centered = user_ratings.subtract(avg_ratings, axis=0)
    # Fill in the missing data with 0s
    user_ratings_normed = user_ratings_centered.fillna(0)

    return user_ratings_normed

def prep_user_ratings(movie_ratings):
    movie_ratings = movie_ratings[:1000]
    user_ratings = movie_ratings.pivot(index='userId', columns='movie_title', values='rating')

    return user_ratings

def prep_user_ratings_centered(movie_ratings):
    user_ratings = prep_user_ratings(movie_ratings)
    user_ratings_centered = fill_nan_ratings(user_ratings)

    return user_ratings_centered

def prep_movie_ratings_centered(user_ratings_centered):
    movie_ratings = user_ratings_centered.T

    return movie_ratings



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
    # Generate a table of titles and plots. 
    # Also preprocess the plots with nlp
    
    movie_plots = plots[["Title", "Plot"]]
    movie_plots.columns = ['title', 'plot']
    movie_plots = movie_plots.set_index('title')
    movie_plots.drop_duplicates(keep='first', inplace=True)
    movie_plots["plot_nlp"] = movie_plots["plot"].apply(nlp_prep)

    return movie_plots



def tfidf_transform(movie_plots):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Get the TF-IDF transformation of the data
    vectorizer = TfidfVectorizer()
    movie_plots = movie_plots[:1000] 
    vec_data = vectorizer.fit_transform(movie_plots["plot_nlp"])

    tfidf_df = pd.DataFrame(vec_data.toarray(), columns=vectorizer.get_feature_names())
    tfidf_df.index = movie_plots.index

    return tfidf_df


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