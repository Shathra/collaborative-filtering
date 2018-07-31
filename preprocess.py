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

def scale_sparse_matrix_rows(s, lowval=0, highval=1):
    d = s.data

    lens = s.getnnz(axis=1)
    idx = np.r_[0,lens[:-1].cumsum()]

    maxs = np.maximum.reduceat(d, idx)
    mins = np.minimum.reduceat(d, idx)

    minsr = np.repeat(mins, lens)
    maxsr = np.repeat(maxs, lens)

    D = highval - lowval
    scaled_01_vals = (d - minsr)/(maxsr - minsr)
    d[:] = scaled_01_vals*D + lowval

user_item_matrix_normalized = user_item_matrix
scale_sparse_matrix_rows(user_item_matrix_normalized, -1, 1)
scipy.sparse.save_npz("user_item_matrix_normalized", user_item_matrix_normalized)

"""
user_item_matrix = user_item_matrix.todense()
# Normalize each row separate
max_arr = user_item_matrix.max(axis=1)
min_arr = np.where(user_item_matrix == 0, user_item_matrix.max(), user_item_matrix).min(axis=1)
row_idx, col_idx = user_item_matrix.nonzero()
user_item_matrix_normalized = scipy.sparse.csr_matrix((no_of_user, no_of_item))
print(row_idx.shape)

c = 0
for row, col in zip(row_idx, col_idx):
	element = user_item_matrix[row, col]
	user_item_matrix_normalized[row, col] = 2 * ((element - min_arr[row]) / (max_arr[row] - min_arr[row])) - 1
	c += 1
	if c % 1000 == 0:
		print(c)
scipy.sparse.save_npz("user_item_matrix_normalized", user_item_matrix_normalized)
"""