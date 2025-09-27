# No external libraries are allowed to be imported in this file
import pandas as pd
import numpy as np


# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    Compute cosine similarity between users (axis=0) or items (axis=1),
    returning a dict[label] -> list[(other_label, similarity)] for the top-k.
    Missing ratings (NaN) are NOT treated as zeros in the math; dimensions
    with no overlap are ignored (intersection cosine).
    """

    similarity_dict = {}

    # Transposing the matrix to manage axis rotation
    ratings = matrix if axis == 0 else matrix.T
    labels = ratings.index.to_numpy()

    ratings_matrix = ratings.to_numpy()
    no_of_labels = ratings_matrix.shape[0]

    # The mask is to filter missing ratings
    is_rated_mask = (~np.isnan(ratings_matrix)).astype(float)
    ratings_zeroed = np.nan_to_num(ratings_matrix, nan=0.0)

    # Numerator
    matrix_transpose_dot = ratings_zeroed @ ratings_zeroed.T

    # Mask filters missing ratings from the squared sum
    ratings_matrix_squared = ratings_zeroed * ratings_zeroed
    a_squared_sum = ratings_matrix_squared @ is_rated_mask.T
    b_squared_sum = a_squared_sum.T
    denominator = np.sqrt(a_squared_sum * b_squared_sum)

    # Contains the cosine similarities for each label pair --> size(no_of_labels, no_of_labels)
    cosine_similarity_matrix = np.divide(
        matrix_transpose_dot, denominator,
        out=np.zeros_like(matrix_transpose_dot, dtype=float),
        where=denominator != 0
    )

    # -inf for diagonal ratings ensures non-max
    np.fill_diagonal(cosine_similarity_matrix, -np.inf)

    # Build top-k indices and cosine similarity scores
    top_k_most_similar = np.argsort(-cosine_similarity_matrix, axis=1)[:, :k]
    top_k_similarities = np.take_along_axis(cosine_similarity_matrix, top_k_most_similar, axis=1)

    for i in range(no_of_labels):
        # Create the desired list of tuples and add as the value to the axis label
        similar_neighbours = [
            (labels[j], float(top_k_similarities[i, t]))
            for t, j in enumerate(top_k_most_similar[i])
        ]
        similarity_dict[labels[i]] = similar_neighbours

    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # Check user_id and movie_id validity
    if user_id not in user_item_matrix.index or movie_id not in user_item_matrix.columns:
        return np.nan

    # Get top-k neighbours
    neighbours = user_similarity.get(user_id, [])[:k]

    numerator = 0
    denominator = 0

    for neighbour_id, sim in neighbours:
        rating = user_item_matrix.at[neighbour_id, movie_id]

        # Evaluate rating only if it exists
        if not np.isnan(rating):
            numerator += sim * rating
            denominator += sim

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie 
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based 
        collaborative filtering
    """
    # Check user_id and movie_id validity
    if user_id not in user_item_matrix.index or movie_id not in user_item_matrix.columns:
        return np.nan

    # Get top-k neighbours
    neighbours = item_similarity.get(movie_id, [])[:k]

    numerator = 0
    denominator = 0

    for neighbour_id, sim in neighbours:
        rating = user_item_matrix.at[user_id, neighbour_id] if user_id in user_item_matrix.index else np.nan

        # Evaluate rating only if it exists
        if not np.isnan(rating):
            numerator += sim * rating
            denominator += sim

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    n_users, n_items = utility_matrix.shape

    # Generating random user and item matrices (weights)
    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)

    rated = [(u, i) for u in range(n_users) for i in range(n_items)
             if not np.isnan(utility_matrix[u, i])]

    for step in range(n_steps):

        # Learning only from existing ratings
        for u, i in rated:
            r_ui = utility_matrix[u, i]

            pred = float(user_matrix[u] @ item_matrix[i])
            err = r_ui - pred

            u_vec = user_matrix[u]
            v_vec = item_matrix[i]

            # Using the regularization formula as suggested to update the weights
            user_matrix[u] = u_vec + learning_rate * (err * v_vec - regularization * u_vec)
            item_matrix[i] = v_vec + learning_rate * (err * u_vec - regularization * v_vec)

    return user_matrix, item_matrix


if __name__ == "__main__":
    path = "u.data"
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index='UserID',
        columns='MovieID',
        values='Rating'
    )
    print(df)

    # You can use this section for testing the similarity_matrix function:
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3, []))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10, []))
    #

    # You can use this section for testing the user_based_cf and the
    # item_based_cf functions: Return the predicted ratings assigned by user
    # 13 to movie 100:
    user_id = 13
    movie_id = 100

    u_predicted_rating = user_based_cf(
        user_id,
        movie_id,
        user_similarity_matrix,
        user_item_matrix=df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id,
        item_similarity_matrix,
        user_item_matrix=df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
