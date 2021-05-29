from itertools import permutations
import pandas as pd
from similarities import jaccard_similarity_big_data, calc_cosine_similarity, cosine_similarity_user_prof
from prep import user_prof_tfidf

class PopularityRecommender:

    def __init__(self, ratings):
        self.ratings = ratings

    def recommendations(self, N=10):
        recommendations = self.ratings.drop_duplicates(subset='movieId', keep='first').reset_index() \
                                                    .sort_values('popularity', ascending=False)
        N_recommendations = recommendations[['movie_title', 'popularity']][:N].reset_index(drop=True)
        return N_recommendations


class AvgRankingRecommender:
    def __init__(self, ratings):
        self.ratings = ratings


    def find_popular_movies(self):
        # Recommend movies that appear more than 50 times in the dataset (popular movies)
        movie_popularity = self.ratings['movie_title'].value_counts()
        popular_movies = movie_popularity[movie_popularity > 50].index
        popular_movie_ratings =  self.ratings[self.ratings['movie_title'].isin(popular_movies)]

        return popular_movie_ratings

    def recommendations(self, N=10):
        popular_movies_average_rankings = self.find_popular_movies()[['movie_title', 'rating']].groupby('movie_title').mean().reset_index()
        popular_movies_average_rankings.columns = ['movie_title', 'avg_rating']
        recommendations = popular_movies_average_rankings.sort_values(by='avg_rating', ascending=False)
        N_recommendations = recommendations[['movie_title', 'avg_rating']][:N].reset_index(drop=True)
        
        return N_recommendations

class PairRecommender:
    def __init__(self, ratings):
        self.ratings = ratings

    def create_pairs(self):
        def find_movie_pairs(x):
            pairs = pd.DataFrame(list(permutations(x.values, 2)),
                                columns=['movie_a', 'movie_b'])
            return pairs

        movie_combinations = self.ratings.groupby('userId')['movie_title'].apply(find_movie_pairs)
        combination_counts = movie_combinations.groupby(['movie_a', 'movie_b']).size().reset_index()
        combination_counts.columns = ['movie_a', 'movie_b', 'pairs_num']
        
        return combination_counts

    def recommendations(self, last_movie_watched, N=10):
        recommendations = self.create_pairs().sort_values('pairs_num', ascending=False)
        recommendations = recommendations[recommendations['movie_a'] == last_movie_watched]

        N_recommendations = recommendations[['movie_b', 'pairs_num']][:N].reset_index(drop=True)

        return N_recommendations

# Content-based recommender, content=genres
class GenreBasedRecommender:
    def __init__(self, movie_genres):
        self.movie_genres = movie_genres

    def recommendations(self, last_movie_watched, N=10):
        target_movie = self.movie_genres.loc[last_movie_watched]
        recommendations = jaccard_similarity_big_data(self.movie_genres, target_movie)
        recommendations = recommendations[recommendations['title'] != last_movie_watched]

        N_recommendations = recommendations.sort_values('jaccard_similarity', ascending=False)[:N]

        return N_recommendations


# Content-based recommender, content=plots
class PlotBasedRecommender():
    def __init__(self, tfidf_movie_plots):
        self.tfidf_movie_plots = tfidf_movie_plots

    def recommendations(self, last_movie_watched, N=10):
        cosine_sim_df = calc_cosine_similarity(self.tfidf_movie_plots)
        N_recommendations = cosine_sim_df.loc[last_movie_watched]
        N_recommendations = N_recommendations.sort_values(ascending=False)[1:(N+1)]
        
        return N_recommendations


class UserProfileRecommender():
    def __init__(self, tfidf_movie_plots):
        self.tfidf_movie_plots = tfidf_movie_plots

    def recommendations(self, movies_enjoyed, N=10):
        user_profile = user_prof_tfidf(self.tfidf_movie_plots, movies_enjoyed)
        similarity_df = cosine_similarity_user_prof(self.tfidf_movie_plots, user_profile, movies_enjoyed)
        sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False)

        return sorted_similarity_df.head(N)