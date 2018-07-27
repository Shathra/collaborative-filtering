import numpy as np
import pandas as pd
import scipy.sparse

data_filepath = "data/ratings.csv"

df = pd.read_csv(data_filepath)

user_arr = np.unique(df["userId"].values)

# Movies ids in rating column are not consecutive, they should be remapped
movie_arr = np.unique(df["movieId"].values)
movie_arr.sort()

movie_id_to_order = np.zeros(movie_arr.max() + 1, dtype=int)
movie_id_to_order[movie_arr] = range(movie_arr.shape[0])

no_of_user = user_arr.shape[0]
no_of_item = movie_arr.shape[0]
user_item_matrix = scipy.sparse.csr_matrix((no_of_user, no_of_item))
df_by_user = df.groupby("userId")
for user_id, group in df_by_user:
    user_item_matrix[user_id - 1, movie_id_to_order[group.movieId.values]] = group.rating.values

scipy.sparse.save_npz("user_item_matrix", user_item_matrix)