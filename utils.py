import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import numpy as np


def calc_df_sparsity(df):
    # Count the occupied cells
    occupied_cell_count = df.isnull().values.sum()

    # Count all cells
    cell_count = df.size

    sparsity = occupied_cell_count / cell_count
    return sparsity


def viz_df(df):
    # Count the occupied cells per column
    occupied_count = df.notnull().sum()

    # Sort the resulting series from low to high
    sorted_occupied_count = occupied_count.sort_values()

    # Plot a histogram of the values in sorted_occupied_count
    sorted_occupied_count.hist()
    plt.show()


def center_ratings_data(user_ratings_df):
    # Get the average rating for each user
    avg_ratings = user_ratings_df.mean(axis=1)

    # Center each user's ratings around 0
    user_ratings_centered = user_ratings_df.sub(avg_ratings, axis=0)

    # Fill in all missing values with 0s
    user_ratings_centered.fillna(0, inplace=True)

    return user_ratings_centered, avg_ratings


def decompose_matrix(user_ratings_df):
    # Decompose the matrix
    user_ratings_centered, avg_ratings = center_ratings_data(user_ratings_df)
    U, sigma, Vt = svds(user_ratings_centered)

    # Convert sigma into a diagonal matrix
    sigma = np.diag(sigma)

    # Dot product of U and sigma
    U_sigma = np.dot(U, sigma)

    # Dot product of result and Vt
    U_sigma_Vt = np.dot(U_sigma, Vt)

    # Add back on the row means contained in avg_ratings
    uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)

    # Create DataFrame of the results
    pred_ratings_df = pd.DataFrame(uncentered_ratings,
                                   index=user_ratings_df.index,
                                   columns=user_ratings_df.columns
                                   )
    return pred_ratings_df
