"""
content_based.py
----------------
Two content-based and matrix factorization approaches:

1. ContentBasedRecommender
   - TF-IDF on genre + year features
   - Cosine similarity between movie feature vectors
   - Movie-to-movie recommendations (no user history needed)

2. SVDRecommender  (Matrix Factorization)
   - Truncated SVD decomposes the user-item matrix into latent factors
   - Captures hidden patterns (e.g. "user likes cerebral sci-fi")
   - Fills in missing ratings → recommends highest predicted-rating items
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


# ══════════════════════════════════════════════════════════════════════════════
# CONTENT-BASED RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

class ContentBasedRecommender:
    """
    Recommends movies similar to a given movie based on content features:
    - Genre tags (weighted via TF-IDF)
    - Release year (normalised)

    This approach requires no user history — ideal for cold-start items.
    """

    def __init__(self):
        self.movies: pd.DataFrame | None = None
        self.similarity_df: pd.DataFrame | None = None
        self.tfidf_matrix = None

    def _build_feature_string(self, row: pd.Series) -> str:
        """
        Combine genres and year into a single weighted text string.
        Genres are repeated to boost their TF-IDF weight.
        """
        genres = " ".join(
            str(row.get("genres", "")).replace("|", " ").split()
        ) * 3   # repeat genres to upweight vs year
        year = str(row.get("year", ""))
        return f"{genres} {year}".strip()

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build TF-IDF feature matrix and cosine similarity matrix.

        Parameters
        ----------
        movies : DataFrame with 'movieId', 'title', 'genres', 'year' columns
        """
        self.movies = movies.set_index("movieId").copy()

        # Build feature strings
        feature_strings = self.movies.apply(self._build_feature_string, axis=1)

        # TF-IDF vectorise
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None,
        )
        self.tfidf_matrix = tfidf.fit_transform(feature_strings)

        # Cosine similarity between all movies
        sim_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.similarity_df = pd.DataFrame(
            sim_matrix,
            index=self.movies.index,
            columns=self.movies.index,
        )
        print(f"[INFO] ContentBased: similarity matrix "
              f"{sim_matrix.shape}, features: {self.tfidf_matrix.shape[1]}")
        return self

    def get_similar_movies(self, movie_id: int,
                           top_n: int = 10) -> pd.DataFrame:
        """
        Return top-N movies most similar to the given movie (by content).

        Returns
        -------
        DataFrame with movieId, title, genres, year, similarity_score
        """
        if movie_id not in self.similarity_df.index:
            return pd.DataFrame()

        scores = (
            self.similarity_df[movie_id]
            .drop(movie_id)
            .sort_values(ascending=False)
            .head(top_n)
        )
        result = scores.reset_index()
        result.columns = ["movieId", "similarity_score"]
        result = result.merge(
            self.movies.reset_index()[["movieId", "title", "genres", "year"]],
            on="movieId",
            how="left",
        )
        result["similarity_score"] = result["similarity_score"].round(4)
        return result[["movieId", "title", "genres", "year", "similarity_score"]]

    def search_by_title(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """
        Fuzzy-search movies by title and return the top-N matches.
        """
        query = query.lower()
        titles = self.movies["title"].str.lower()
        # Exact prefix match first, then substring
        exact   = titles[titles.str.startswith(query)]
        partial = titles[titles.str.contains(query, na=False)]
        combined = pd.concat([exact, partial]).drop_duplicates()
        ids = combined.index[:top_n].tolist()
        return self.movies.loc[ids].reset_index()[["movieId", "title", "genres", "year"]]


# ══════════════════════════════════════════════════════════════════════════════
# SVD / MATRIX FACTORIZATION RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

class SVDRecommender:
    """
    Matrix Factorization via Truncated SVD.

    How it works
    ------------
    The user-item rating matrix R (users × movies) is decomposed into:
        R ≈ U × Σ × Vᵀ
    where U contains user latent factors and V contains item latent factors.
    The dot product U × Vᵀ fills in missing entries (predicted ratings).

    This captures latent tastes like "user likes cerebral sci-fi" without
    needing explicit genre labels.
    """

    def __init__(self, n_components: int = 50):
        self.n_components  = n_components
        self.svd           = TruncatedSVD(n_components=n_components,
                                          random_state=42)
        self.scaler        = MinMaxScaler(feature_range=(1, 5))
        self.matrix: pd.DataFrame | None = None
        self.predicted_df:  pd.DataFrame | None = None
        self.U = None     # user factors
        self.V = None     # item factors

    def fit(self, matrix: pd.DataFrame) -> "SVDRecommender":
        """
        Decompose the user-item matrix into latent factors.

        Parameters
        ----------
        matrix : DataFrame (users × movies), NaN = unrated (will be filled with 0)
        """
        self.matrix = matrix
        R = matrix.fillna(0).values  # dense matrix with 0s for missing

        # Decompose: U (users × k), Σ embedded in U, V (movies × k)
        self.U  = self.svd.fit_transform(R)    # shape: (n_users, n_components)
        self.V  = self.svd.components_.T       # shape: (n_movies, n_components)

        # Reconstruct predicted ratings matrix
        R_hat = self.U @ self.V.T              # (n_users × n_movies)

        # Scale predicted values to [1, 5] range
        R_hat_scaled = self.scaler.fit_transform(R_hat)

        self.predicted_df = pd.DataFrame(
            R_hat_scaled,
            index=matrix.index,
            columns=matrix.columns,
        )

        explained = self.svd.explained_variance_ratio_.sum()
        print(f"[INFO] SVD: {n_components} components, "
              f"explained variance: {explained:.1%}")
        return self

    def recommend_for_user(self, user_id: int,
                           top_n: int = 10,
                           exclude_seen: bool = True) -> pd.DataFrame:
        """
        Recommend movies by highest predicted rating for a given user.

        Parameters
        ----------
        exclude_seen : if True, remove movies the user has already rated
        """
        if self.predicted_df is None or user_id not in self.predicted_df.index:
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        pred_row = self.predicted_df.loc[user_id].copy()

        if exclude_seen:
            seen_ids = self.matrix.loc[user_id].dropna().index
            pred_row = pred_row.drop(seen_ids, errors="ignore")

        top = (
            pred_row.sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        top.columns = ["movieId", "predicted_rating"]
        top["predicted_rating"] = top["predicted_rating"].round(4)
        return top

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Find movies similar to a given movie based on latent factor similarity.
        Uses cosine similarity of item (V) vectors.
        """
        if movie_id not in self.matrix.columns:
            return pd.DataFrame(columns=["movieId", "similarity_score"])

        movie_cols = list(self.matrix.columns)
        mid_idx = movie_cols.index(movie_id)
        movie_vec = self.V[mid_idx].reshape(1, -1)

        sims = cosine_similarity(movie_vec, self.V)[0]
        sim_series = pd.Series(sims, index=self.matrix.columns)
        top = (
            sim_series.drop(movie_id)
            .sort_values(ascending=False)
            .head(top_n)
        )
        result = top.reset_index()
        result.columns = ["movieId", "similarity_score"]
        result["similarity_score"] = result["similarity_score"].round(4)
        return result


# Make n_components accessible as module-level variable for SVDRecommender init
n_components = 50
