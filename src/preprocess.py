"""
preprocess.py
-------------
Loads movies and ratings, builds user-item matrix,
and prepares data structures used by all recommender models.
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ── Loading ───────────────────────────────────────────────────────────────────

def load_ratings(path: str = "data/ratings.csv") -> pd.DataFrame:
    """Load user-movie ratings CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"userId", "movieId", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns: {required}. Got: {set(df.columns)}")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df.dropna(subset=["rating"], inplace=True)
    df["rating"] = df["rating"].clip(1, 5)
    print(f"[INFO] Ratings: {len(df):,} | Users: {df['userId'].nunique()} "
          f"| Movies: {df['movieId'].nunique()}")
    return df


def load_movies(path: str = "data/movies.csv") -> pd.DataFrame:
    """Load movies metadata CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Movies file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"[INFO] Movies: {len(df)}")
    return df


# ── User-item matrix ──────────────────────────────────────────────────────────

def build_user_item_matrix(ratings: pd.DataFrame) \
        -> tuple[pd.DataFrame, dict, dict]:
    """
    Pivot ratings into a (users × movies) matrix.
    NaN = movie not rated by that user.

    Returns
    -------
    matrix        : DataFrame (userId index, movieId columns)
    user_to_idx   : {userId: row_index}
    movie_to_idx  : {movieId: col_index}
    """
    matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    user_to_idx  = {uid: i for i, uid in enumerate(matrix.index)}
    movie_to_idx = {mid: i for i, mid in enumerate(matrix.columns)}
    return matrix, user_to_idx, movie_to_idx


def build_sparse_matrix(matrix: pd.DataFrame) -> csr_matrix:
    """
    Convert the dense user-item matrix to a sparse CSR matrix
    (fill NaN → 0) for memory-efficient similarity computation.
    """
    return csr_matrix(matrix.fillna(0).values)


# ── Genre / content features ──────────────────────────────────────────────────

def build_genre_matrix(movies: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the pipe-separated genres column.
    Returns a DataFrame indexed by movieId.
    """
    if "genres" not in movies.columns:
        return pd.DataFrame(index=movies["movieId"])

    # Explode pipe-separated genres into binary columns
    genres_split = movies["genres"].str.split("|").apply(
        lambda lst: [g.strip() for g in (lst if isinstance(lst, list) else [])]
    )
    all_genres = sorted({g for row in genres_split for g in row})

    genre_matrix = pd.DataFrame(
        [[1 if g in row else 0 for g in all_genres] for row in genres_split],
        columns=all_genres,
        index=movies["movieId"],
    )
    return genre_matrix


# ── Rating normalisation ──────────────────────────────────────────────────────

def normalise_ratings(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-centre each user's ratings (subtract row mean).
    Used for better cosine similarity in collaborative filtering.
    """
    row_means = matrix.mean(axis=1)
    return matrix.sub(row_means, axis=0).fillna(0)


# ── Train / test split ────────────────────────────────────────────────────────

def leave_one_out_split(ratings: pd.DataFrame,
                         random_state: int = 42) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user, hold out their most recently-added rating as the test item.
    This mirrors how real recommender evaluation works.
    """
    rng = np.random.default_rng(random_state)
    test_rows = []
    train_rows = []

    for uid, group in ratings.groupby("userId"):
        if len(group) < 3:
            # Not enough ratings to evaluate — keep all in train
            train_rows.append(group)
            continue
        # Hold out one random rating per user
        test_idx = rng.choice(group.index)
        test_rows.append(group.loc[[test_idx]])
        train_rows.append(group.drop(test_idx))

    train = pd.concat(train_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)
    print(f"[INFO] Train ratings: {len(train):,} | Test ratings: {len(test):,}")
    return train, test


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess(ratings_path: str = "data/ratings.csv",
               movies_path:  str = "data/movies.csv") -> dict:
    """
    Load, clean, and structure all data.

    Returns a dict with:
        ratings, movies, matrix, sparse_matrix,
        user_to_idx, movie_to_idx, genre_matrix,
        norm_matrix, train_ratings, test_ratings
    """
    ratings = load_ratings(ratings_path)
    movies  = load_movies(movies_path)

    # Merge to keep only movies with metadata
    valid_ids = set(movies["movieId"])
    ratings   = ratings[ratings["movieId"].isin(valid_ids)].copy()

    matrix, user_to_idx, movie_to_idx = build_user_item_matrix(ratings)
    sparse_mat   = build_sparse_matrix(matrix)
    norm_matrix  = normalise_ratings(matrix)
    genre_matrix = build_genre_matrix(movies)

    train_ratings, test_ratings = leave_one_out_split(ratings)

    print(f"[INFO] Matrix shape: {matrix.shape} "
          f"(sparsity: {1 - ratings.shape[0] / (matrix.shape[0]*matrix.shape[1]):.1%})")

    return {
        "ratings":       ratings,
        "movies":        movies,
        "matrix":        matrix,
        "sparse_matrix": sparse_mat,
        "user_to_idx":   user_to_idx,
        "movie_to_idx":  movie_to_idx,
        "genre_matrix":  genre_matrix,
        "norm_matrix":   norm_matrix,
        "train_ratings": train_ratings,
        "test_ratings":  test_ratings,
    }
