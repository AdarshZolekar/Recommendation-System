"""
utils.py
--------
Shared helpers:
  - Artifact persistence (pickle)
  - HybridRecommender: blends CF + content-based scores
  - Movie info lookup helpers
  - Genre colour mapping for the Streamlit UI
"""

import os
import pickle
import numpy as np
import pandas as pd


# ── Artifact I/O ──────────────────────────────────────────────────────────────

def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve(relative: str) -> str:
    return os.path.join(repo_root(), relative)


def save_pickle(obj, path: str) -> None:
    abs_path = resolve(path) if not os.path.isabs(path) else path
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved → {path}")


def load_pickle(path: str):
    abs_path = resolve(path) if not os.path.isabs(path) else path
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Artifact not found: {abs_path}\n"
            "Run `python main.py` first to train and save models."
        )
    with open(abs_path, "rb") as f:
        return pickle.load(f)


# ── Hybrid recommender ────────────────────────────────────────────────────────

class HybridRecommender:
    """
    Blends Item-Based CF + SVD predictions using a weighted average.

    α = weight on collaborative filtering (0–1)
    (1-α) = weight on SVD/content signal

    Hybrid systems dominate real-world applications (Netflix, Spotify)
    because they mitigate weaknesses of each individual approach:
      - CF fails for new items (cold-start) → content rescues it
      - Content fails for niche tastes → CF rescues it
    """

    def __init__(self, item_cf, svd, alpha: float = 0.6):
        """
        Parameters
        ----------
        item_cf : fitted ItemBasedCF
        svd     : fitted SVDRecommender
        alpha   : weight for CF scores (0=pure SVD, 1=pure CF)
        """
        self.item_cf = item_cf
        self.svd     = svd
        self.alpha   = alpha

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Blend CF and SVD predicted ratings and return top-N.
        """
        cf_recs  = self.item_cf.recommend_for_user(user_id, top_n=50)
        svd_recs = self.svd.recommend_for_user(user_id, top_n=50)

        if cf_recs.empty and svd_recs.empty:
            return pd.DataFrame(columns=["movieId", "hybrid_score"])

        # Normalise each model's scores to [0, 1]
        def normalise(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn + 1e-9)

        cf_scores  = cf_recs.set_index("movieId")["predicted_rating"]
        svd_scores = svd_recs.set_index("movieId")["predicted_rating"]

        cf_norm  = normalise(cf_scores)
        svd_norm = normalise(svd_scores)

        # Combine on union of movie IDs
        all_ids = cf_norm.index.union(svd_norm.index)
        cf_filled  = cf_norm.reindex(all_ids).fillna(0)
        svd_filled = svd_norm.reindex(all_ids).fillna(0)

        hybrid = self.alpha * cf_filled + (1 - self.alpha) * svd_filled
        top = hybrid.sort_values(ascending=False).head(top_n).reset_index()
        top.columns = ["movieId", "hybrid_score"]
        top["hybrid_score"] = top["hybrid_score"].round(4)
        return top


# ── Movie info helpers ────────────────────────────────────────────────────────

def get_movie_info(movie_id: int, movies: pd.DataFrame) -> dict:
    """Return a movie's metadata as a dict, or empty dict if not found."""
    row = movies[movies["movieId"] == movie_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def enrich_recommendations(recs: pd.DataFrame, movies: pd.DataFrame,
                            score_col: str = "predicted_rating") -> pd.DataFrame:
    """
    Merge recommendation scores with movie metadata.

    Returns DataFrame with: movieId, title, genres, year, score_col
    """
    enriched = recs.merge(
        movies[["movieId", "title", "genres", "year"]],
        on="movieId",
        how="left",
    )
    cols = ["movieId", "title", "genres", "year", score_col]
    cols = [c for c in cols if c in enriched.columns]
    return enriched[cols].reset_index(drop=True)


def get_user_rated_movies(user_id: int, ratings: pd.DataFrame,
                           movies: pd.DataFrame,
                           min_rating: float = 0.0) -> pd.DataFrame:
    """Return all movies rated by a user, sorted by rating descending."""
    user_ratings = ratings[
        (ratings["userId"] == user_id) &
        (ratings["rating"] >= min_rating)
    ].copy()
    result = user_ratings.merge(movies[["movieId", "title", "genres", "year"]],
                                 on="movieId", how="left")
    return result.sort_values("rating", ascending=False).reset_index(drop=True)


# ── Genre display config ──────────────────────────────────────────────────────

GENRE_COLORS = {
    "Action":      "#ef4444",
    "Adventure":   "#f97316",
    "Animation":   "#eab308",
    "Biography":   "#84cc16",
    "Comedy":      "#22c55e",
    "Crime":       "#14b8a6",
    "Drama":       "#3b82f6",
    "Fantasy":     "#8b5cf6",
    "History":     "#a78bfa",
    "Horror":      "#dc2626",
    "Music":       "#ec4899",
    "Mystery":     "#6366f1",
    "Romance":     "#f43f5e",
    "Sci-Fi":      "#06b6d4",
    "Thriller":    "#0ea5e9",
    "War":         "#64748b",
    "Western":     "#d97706",
}

DEFAULT_COLOR = "#94a3b8"


def genre_badge_html(genres_str: str) -> str:
    """Return HTML badges for each genre in a pipe-separated string."""
    if not genres_str or pd.isna(genres_str):
        return ""
    badges = []
    for g in str(genres_str).split("|"):
        g = g.strip()
        color = GENRE_COLORS.get(g, DEFAULT_COLOR)
        badges.append(
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:12px;font-size:0.72rem;font-weight:600;'
            f'margin-right:4px">{g}</span>'
        )
    return "".join(badges)
