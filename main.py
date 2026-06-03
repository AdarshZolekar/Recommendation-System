"""
main.py
-------
Full recommendation system pipeline.
Trains all models on the train split so evaluation is honest.

Usage:
    python main.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess      import preprocess, build_user_item_matrix, normalise_ratings
from collaborative   import ItemBasedCF, UserBasedCF
from content_based   import ContentBasedRecommender, SVDRecommender
from evaluate        import evaluate_recommender, print_metrics, print_comparison
from utils           import (save_pickle, HybridRecommender,
                              enrich_recommendations, get_user_rated_movies)

# ── Config ─────────────────────────────────────────────────────────────────

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH  = "data/movies.csv"
K            = 10

MODEL_PATHS = {
    "item_cf":  "models/item_cf.pkl",
    "user_cf":  "models/user_cf.pkl",
    "content":  "models/content_based.pkl",
    "svd":      "models/svd_model.pkl",
    "hybrid":   "models/hybrid.pkl",
    "data":     "models/data_bundle.pkl",
}

# ── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "=" * 60)
    print("   RECOMMENDATION SYSTEM — PIPELINE")
    print("=" * 60)

    # Step 1: Preprocess
    print("\n[STEP 1] Loading and preprocessing data...")
    data = preprocess(RATINGS_PATH, MOVIES_PATH)
    ratings, movies = data["ratings"], data["movies"]
    os.makedirs("models", exist_ok=True)
    save_pickle(data, MODEL_PATHS["data"])

    # Build train-only matrix — models must NOT see test items
    train_matrix, _, _ = build_user_item_matrix(data["train_ratings"])
    train_norm          = normalise_ratings(train_matrix)

    # Step 2: Train all models
    print("\n[STEP 2] Training models...")

    item_cf = ItemBasedCF().fit(train_matrix)
    save_pickle(item_cf, MODEL_PATHS["item_cf"])

    user_cf = UserBasedCF(k=15).fit(train_matrix, train_norm)
    save_pickle(user_cf, MODEL_PATHS["user_cf"])

    content = ContentBasedRecommender().fit(movies)
    save_pickle(content, MODEL_PATHS["content"])

    svd = SVDRecommender(n_components=30).fit(train_matrix)
    save_pickle(svd, MODEL_PATHS["svd"])

    hybrid = HybridRecommender(item_cf=item_cf, svd=svd, alpha=0.6)
    save_pickle(hybrid, MODEL_PATHS["hybrid"])

    # Step 3: Evaluate on test split
    print(f"\n[STEP 3] Evaluating models at K={K}...")
    train_r = data["train_ratings"]
    test_r  = data["test_ratings"]

    def item_cf_fn(uid, n): return item_cf.recommend_for_user(uid, n)["movieId"].tolist()
    def user_cf_fn(uid, n): return user_cf.recommend_for_user(uid, n)["movieId"].tolist()
    def svd_fn(uid, n):     return svd.recommend_for_user(uid, n)["movieId"].tolist()
    def hybrid_fn(uid, n):  return hybrid.recommend_for_user(uid, n)["movieId"].tolist()

    eval_results = {}
    for name, fn in [
        ("Item-Based CF", item_cf_fn),
        ("User-Based CF", user_cf_fn),
        ("SVD",           svd_fn),
        ("Hybrid",        hybrid_fn),
    ]:
        print(f"\n  Evaluating {name}...")
        m = evaluate_recommender(fn, test_r, train_r, k=K)
        print_metrics(m, model_name=name, k=K)
        eval_results[name] = m

    print_comparison(eval_results, k=K)

    # Step 4: Live demo
    print("\n[STEP 4] Recommendation demos...")

    sample_mid = 3  # The Dark Knight
    title = movies[movies["movieId"] == sample_mid]["title"].values[0]
    print(f"\n  Content-based similar to '{title}':")
    sims = content.get_similar_movies(sample_mid, top_n=5)
    for _, row in sims.iterrows():
        print(f"    [{row['similarity_score']:.2f}] {row['title']} ({row['year']}) — {row['genres']}")

    sample_uid = 1
    liked = get_user_rated_movies(sample_uid, ratings, movies, min_rating=4.0)
    print(f"\n  Hybrid recommendations for User {sample_uid}:")
    print(f"    User liked: {', '.join(liked['title'].head(3).tolist())}")
    recs = hybrid.recommend_for_user(sample_uid, top_n=5)
    recs = enrich_recommendations(recs, movies, score_col="hybrid_score")
    for _, row in recs.iterrows():
        print(f"    [{row['hybrid_score']:.2f}] {row['title']} ({row.get('year','')}) — {row.get('genres','')}")

    print("\n[STEP 5] Pipeline complete. All models saved to models/")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    run_pipeline()
