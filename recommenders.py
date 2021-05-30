from itertools import permutations
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from similarities import jaccard_similarity_big_data, calc_cosine_similarity, cosine_similarity_user_prof
from prep import user_prof_tfidf
from utils import decompose_matrix


class PopularityRecommender:

    def __init__(self, ratings):
        self.ratings = ratings

    def recommendations(self, N=10):
        result = self.ratings.drop_duplicates(subset='movieId', keep='first').reset_index() \
            .sort_values('popularity', ascending=False)
        n_recommendations = result[[
            'movie_title', 'popularity']][:N].reset_index(drop=True)
        return n_recommendations


class AvgRankingRecommender:
    def __init__(self, ratings):
        self.ratings = ratings

    def find_popular_movies(self):
        # Recommend movies that appear more than 50 times in the dataset (popular movies)
        movie_popularity = self.ratings['movie_title'].value_counts()
        popular_movies = movie_popularity[movie_popularity > 50].index
        popular_movie_ratings = self.ratings[self.ratings['movie_title'].isin(
            popular_movies)]

        return popular_movie_ratings

    def recommendations(self, N=10):
        popular_movies_average_rankings = self.find_popular_movies(
        )[['movie_title', 'rating']].groupby('movie_title').mean().reset_index()
        popular_movies_average_rankings.columns = ['movie_title', 'avg_rating']
        result = popular_movies_average_rankings.sort_values(
            by='avg_rating', ascending=False)
        n_recommendations = result[[
            'movie_title', 'avg_rating']][:N].reset_index(drop=True)

        return n_recommendations


class PairRecommender:
    def __init__(self, ratings):
        self.ratings = ratings

    def create_pairs(self):
        def find_movie_pairs(x):
            pairs = pd.DataFrame(list(permutations(x.values, 2)),
                                 columns=['movie_a', 'movie_b'])
            return pairs

        movie_combinations = self.ratings.groupby(
            'userId')['movie_title'].apply(find_movie_pairs)
        combination_counts = movie_combinations.groupby(
            ['movie_a', 'movie_b']).size().reset_index()
        combination_counts.columns = ['movie_a', 'movie_b', 'pairs_num']

        return combination_counts

    def recommendations(self, last_movie_watched, N=10):
        result = self.create_pairs().sort_values('pairs_num', ascending=False)
        result = result[result['movie_a']
                        == last_movie_watched]

        n_recommendations = result[[
            'movie_b', 'pairs_num']][:N].reset_index(drop=True)

        return n_recommendations

# Content-based recommender, content=genres


class GenreBasedRecommender:
    def __init__(self, movie_genres):
        self.movie_genres = movie_genres

    def recommendations(self, last_movie_watched, N=10):
        target_movie = self.movie_genres.loc[last_movie_watched]
        similarities = jaccard_similarity_big_data(
            self.movie_genres, target_movie)
        similarities = similarities[similarities['title']
                                    != last_movie_watched]

        n_recommendations = similarities.sort_values(
            'jaccard_similarity', ascending=False)[:N]

        return n_recommendations


# Content-based recommender, content=plots
class PlotBasedRecommender:
    def __init__(self, tfidf_movie_plots):
        self.tfidf_movie_plots = tfidf_movie_plots

    def recommendations(self, last_movie_watched, N=10):
        cosine_sim_df = calc_cosine_similarity(self.tfidf_movie_plots)
        n_recommendations = cosine_sim_df.loc[last_movie_watched]
        n_recommendations = n_recommendations.sort_values(ascending=False)[
            1:(N+1)]

        return n_recommendations


class UserProfileRecommender:
    def __init__(self, tfidf_movie_plots):
        self.tfidf_movie_plots = tfidf_movie_plots

    def recommendations(self, movies_enjoyed, N=10):
        user_profile = user_prof_tfidf(self.tfidf_movie_plots, movies_enjoyed)
        similarity_df = cosine_similarity_user_prof(
            self.tfidf_movie_plots, user_profile, movies_enjoyed)
        sorted_similarity_df = similarity_df.sort_values(
            by="similarity_score", ascending=False)

        return sorted_similarity_df.head(N)


class CollaborativeFilteringRecommender:
    def __init__(self, movie_ratings_centered):
        self.movie_ratings_centered = movie_ratings_centered

    def recommendations(self, last_movie_watched, N=10):

        cosine_similarity_df = calc_cosine_similarity(
            self.movie_ratings_centered)

        # Find the similarity values for a specific movie
        cosine_similarity_series = cosine_similarity_df.loc[last_movie_watched]

        # Sort these values highest to lowest
        n_recommendations = cosine_similarity_series.sort_values(
            ascending=False)
        n_recommendations.drop(labels=[last_movie_watched], inplace=True)

        return n_recommendations[:N]


class KnnRecommender:

    def __init__(self, user_ratings, centered_user_ratings):
        self.user_ratings = user_ratings
        self.centered_user_ratings = centered_user_ratings

    def knn_model_prediction(self, target_user, target_movie):

       # Drop the column you are trying to predict
        centered_exclude_movie = self.centered_user_ratings.drop(
            target_movie, axis=1)
        # Get the data for the user you are predicting for
        target_user_x = centered_exclude_movie.loc[[target_user]]
        # Get the target data from user_ratings_table
        other_users_y = self.user_ratings[[target_movie]]
        # Get the data for only those that have seen the movie
        other_users_x = centered_exclude_movie[other_users_y.notnull()[
            target_movie]]
        # Remove those that have not seen the movie from the target
        other_users_y.dropna(inplace=True)

        # Instantiate the user KNN model
        knn_model = KNeighborsRegressor(metric='cosine', n_neighbors=1)

        # Fit the model and predict the target user
        knn_model.fit(other_users_x, other_users_y)
        prediction = knn_model.predict(target_user_x)

        return prediction[0].item()

    def recommendations(self, user, N=10):
        # TODO find all unwatched movies / assume all watched movies are also rated
        # TODO apply KNN model to predict the user rating and store for each unwatched movie
        pred_ratings = []
        user_movies = self.user_ratings.loc[1]
        unwatched_movies = user_movies[user_movies.isna()].index.tolist()
        for movie in unwatched_movies:
            pred_rating = self.knn_model_prediction(user, movie)
            pred_ratings.append(pred_rating)

        # TODO order the movie and the predicted ratings
        predictions = pd.DataFrame(
            {'unwatched_movies': unwatched_movies, 'pred_ratings': pred_ratings})

        n_recommendations = predictions.sort_values(
            'pred_ratings', ascending=False)[:N]

        return n_recommendations


class SVDRecomender:
    def __init__(self, user_ratings):
        self.user_ratings = user_ratings

    def recommendations(self, user, N=10):
        predictions = decompose_matrix(self.user_ratings)
        n_recommendations = predictions.loc[user, :].sort_values(ascending=False)[
            :N]

        return n_recommendations
