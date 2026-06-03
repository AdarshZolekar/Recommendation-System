"""
collaborative.py
----------------
Two collaborative filtering strategies:

1. Item-Based CF  (preferred — more stable, scales better)
   - Builds an item × item cosine similarity matrix
   - For a given movie, returns the most similar movies

2. User-Based CF
   - Builds a user × user similarity matrix
   - Finds the K most similar users to a target user
   - Recommends movies those users liked that the target hasn't seen

Both use mean-centred ratings for better similarity estimation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ══════════════════════════════════════════════════════════════════════════════
# ITEM-BASED COLLABORATIVE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

class ItemBasedCF:
    """
    Item-Item Collaborative Filtering using cosine similarity.

    Intuition
    ---------
    If User A and User B both rated Movie X and Movie Y similarly,
    then Movie X and Movie Y are "similar" items. When a new user
    rates Movie X highly, we recommend Movie Y to them.
    """

    def __init__(self):
        self.similarity_df: pd.DataFrame | None = None
        self.matrix: pd.DataFrame | None = None

    def fit(self, matrix: pd.DataFrame) -> "ItemBasedCF":
        """
        Compute item-item cosine similarity from the user-item matrix.

        Parameters
        ----------
        matrix : DataFrame (users × movies), NaN = unrated
        """
        self.matrix = matrix

        # Transpose → movies × users; fill NaN with 0 for similarity calc
        item_matrix = matrix.fillna(0).T
        sim = cosine_similarity(item_matrix)

        self.similarity_df = pd.DataFrame(
            sim,
            index=matrix.columns,
            columns=matrix.columns,
        )
        print(f"[INFO] ItemBasedCF: similarity matrix {sim.shape}")
        return self

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top-N most similar movies to `movie_id`.

        Returns
        -------
        DataFrame with columns: movieId, similarity_score
        """
        if movie_id not in self.similarity_df.index:
            return pd.DataFrame(columns=["movieId", "similarity_score"])

        scores = (
            self.similarity_df[movie_id]
            .drop(movie_id)          # exclude itself
            .sort_values(ascending=False)
            .head(top_n)
        )
        return pd.DataFrame({
            "movieId":          scores.index,
            "similarity_score": scores.values.round(4),
        })

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                           min_similarity: float = 0.1) -> pd.DataFrame:
        """
        Predict unseen movie scores for a user using weighted similarities.

        For each unrated movie m, score(m) = Σ sim(m, rated_i) × rating_i
        normalised by the sum of similarities.
        """
        if user_id not in self.matrix.index:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        user_ratings = self.matrix.loc[user_id].dropna()
        seen_ids     = set(user_ratings.index)
        unseen_ids   = [mid for mid in self.matrix.columns if mid not in seen_ids]

        if not unseen_ids:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        scores = {}
        for mid in unseen_ids:
            # Similarity to all movies the user has rated
            sims = self.similarity_df.loc[mid, user_ratings.index]
            mask = sims >= min_similarity
            if mask.sum() == 0:
                continue
            weighted = (sims[mask] * user_ratings[mask]).sum()
            denom    = sims[mask].abs().sum()
            scores[mid] = weighted / denom if denom > 0 else 0.0

        if not scores:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        result = (
            pd.Series(scores)
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        result.columns = ["movieId", "predicted_rating"]
        result["predicted_rating"] = result["predicted_rating"].round(4)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# USER-BASED COLLABORATIVE FILTERING WITH KNN
# ══════════════════════════════════════════════════════════════════════════════

class UserBasedCF:
    """
    User-User Collaborative Filtering using KNN (cosine distance).

    Intuition
    ---------
    "People like you also liked…"
    Find K users with similar rating histories and recommend
    movies they liked that the target user hasn't seen yet.
    """

    def __init__(self, k: int = 15):
        self.k = k
        self.knn: NearestNeighbors | None = None
        self.matrix: pd.DataFrame | None = None
        self.norm_matrix: pd.DataFrame | None = None

    def fit(self, matrix: pd.DataFrame,
            norm_matrix: pd.DataFrame | None = None) -> "UserBasedCF":
        """
        Fit KNN on the user-item matrix (mean-centred if provided).

        Parameters
        ----------
        matrix      : raw user-item matrix
        norm_matrix : mean-centred user-item matrix (preferred for KNN)
        """
        self.matrix      = matrix
        self.norm_matrix = norm_matrix if norm_matrix is not None else matrix.fillna(0)

        self.knn = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(matrix)),
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        self.knn.fit(self.norm_matrix.fillna(0).values)
        print(f"[INFO] UserBasedCF KNN: k={self.k}, users={len(matrix)}")
        return self

    def _get_similar_users(self, user_id: int) -> list[tuple[int, float]]:
        """Return (user_id, similarity_score) for the K nearest users."""
        if user_id not in self.matrix.index:
            return []

        user_idx = list(self.matrix.index).index(user_id)
        user_vec = self.norm_matrix.iloc[user_idx].fillna(0).values.reshape(1, -1)

        distances, indices = self.knn.kneighbors(user_vec)
        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            uid = self.matrix.index[idx]
            if uid == user_id:
                continue
            sim = 1 - dist  # cosine distance → cosine similarity
            similar.append((uid, round(sim, 4)))
        return similar[:self.k]

    def recommend_for_user(self, user_id: int, top_n: int = 10,
                           min_rating: float = 3.5) -> pd.DataFrame:
        """
        Recommend unseen movies based on what similar users liked.

        Parameters
        ----------
        min_rating : minimum rating by a similar user to count as "liked"
        """
        if user_id not in self.matrix.index:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        seen_ids = set(self.matrix.loc[user_id].dropna().index)
        similar_users = self._get_similar_users(user_id)

        if not similar_users:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        # Aggregate: weighted average predicted rating per unseen movie
        movie_scores: dict[int, list[float]] = {}
        movie_sims:   dict[int, list[float]] = {}

        for sim_uid, sim_score in similar_users:
            sim_ratings = self.matrix.loc[sim_uid].dropna()
            # Only consider movies the similar user liked
            liked = sim_ratings[
                (sim_ratings >= min_rating) &
                (~sim_ratings.index.isin(seen_ids))
            ]
            for mid, rating in liked.items():
                movie_scores.setdefault(mid, []).append(rating * sim_score)
                movie_sims.setdefault(mid, []).append(sim_score)

        if not movie_scores:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        predictions = {
            mid: sum(scores) / sum(movie_sims[mid])
            for mid, scores in movie_scores.items()
            if sum(movie_sims[mid]) > 0
        }

        result = (
            pd.Series(predictions)
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        result.columns = ["movieId", "predicted_rating"]
        result["predicted_rating"] = result["predicted_rating"].round(4)
        return result

    def get_similar_users(self, user_id: int) -> pd.DataFrame:
        """Return a DataFrame of similar users and their similarity scores."""
        similar = self._get_similar_users(user_id)
        return pd.DataFrame(similar, columns=["userId", "similarity_score"])
