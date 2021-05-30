# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from recommenders import SVDRecommender


class RatingsEvaluator():
    def __init__(self, user_ratings):
        self.user_ratings = user_ratings
        self.holdout = self.user_ratings.iloc[:5, :10].values

    def evaluate(self, predictions_df):

        predicted = predictions_df.iloc[:5, :10].values
        # Create a mask of actual_values to only look at the non-missing values in the ground truth
        mask = ~np.isnan(self.holdout)
        # Print the performance of both predictions and compare
        rmse = mean_squared_error(
            self.holdout[mask], predicted[mask])

        return rmse


if __name__ == "__main__":
    user_ratings = pd.read_csv(
        'data/user_ratings.csv', index_col='userId')
    avg_ratings_per_movie = user_ratings.fillna(user_ratings.mean())
    svd_rec = SVDRecommender(user_ratings)
    eval = RatingsEvaluator(user_ratings)
    rmse_avg_rec = eval.evaluate(avg_ratings_per_movie)
    rmse_svd_rec = eval.evaluate(svd_rec.predict())

    print('RMSE for SVD recommender: ', rmse_svd_rec)
    print('RMSE for Avg ranking recommender: ', rmse_avg_rec)
