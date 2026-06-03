"""
evaluate.py
-----------
Standard information-retrieval metrics for recommendation systems.

Metrics implemented
-------------------
Precision@K   — Of the K items recommended, what fraction were relevant?
Recall@K      — Of all relevant items, how many did we catch in the top K?
F1@K          — Harmonic mean of Precision and Recall at K
NDCG@K        — Normalised Discounted Cumulative Gain (rewards highly-ranked hits)
Hit Rate@K    — Did at least one relevant item appear in the top K? (binary)
Coverage      — What fraction of the catalogue was ever recommended?
"""

import numpy as np
import pandas as pd
from typing import Callable


# ── Per-query metrics ─────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-K recommendations."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    DCG@K  = Σ_i  rel_i / log2(i + 2)   for i in 0..k-1
    IDCG@K = best possible DCG (all relevant items at the top)
    NDCG@K = DCG@K / IDCG@K
    """
    top_k = recommended[:k]
    dcg   = sum(
        (1 / np.log2(i + 2)) for i, item in enumerate(top_k) if item in relevant
    )
    idcg  = sum(
        1 / np.log2(i + 2) for i in range(min(len(relevant), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """1 if any relevant item is in the top-K, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_recommender(
    recommend_fn: Callable[[int, int], list[int]],
    test_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    k: int = 10,
    min_test_rating: float = 3.5,
) -> dict:
    """
    Evaluate a recommender over all test users.

    Parameters
    ----------
    recommend_fn    : function(user_id, top_n) → list of movie IDs
    test_ratings    : held-out ratings (one per user from leave-one-out split)
    train_ratings   : training ratings (to exclude seen items)
    k               : evaluation cut-off
    min_test_rating : minimum rating to treat as "relevant"

    Returns
    -------
    dict with mean Precision@K, Recall@K, F1@K, NDCG@K, HitRate@K
    """
    precisions, recalls, f1s, ndcgs, hits = [], [], [], [], []

    # Only evaluate users who appear in both train and test
    train_users = set(train_ratings["userId"].unique())
    test_users  = set(test_ratings["userId"].unique())
    eval_users  = train_users & test_users

    n_skipped = 0
    for uid in eval_users:
        # Relevant items: test movies with rating ≥ threshold
        user_test = test_ratings[
            (test_ratings["userId"] == uid) &
            (test_ratings["rating"] >= min_test_rating)
        ]
        relevant = set(user_test["movieId"].tolist())
        if not relevant:
            n_skipped += 1
            continue

        # Get recommendations
        try:
            recs = recommend_fn(uid, k)
        except Exception:
            n_skipped += 1
            continue

        if not recs:
            n_skipped += 1
            continue

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        f1s.append(f1_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        hits.append(hit_rate_at_k(recs, relevant, k))

    n_evaluated = len(precisions)
    print(f"[EVAL] Evaluated {n_evaluated} users (skipped {n_skipped})")

    if n_evaluated == 0:
        return {m: 0.0 for m in
                [f"precision@{k}", f"recall@{k}", f"f1@{k}", f"ndcg@{k}", f"hit_rate@{k}"]}

    return {
        f"precision@{k}": round(np.mean(precisions), 4),
        f"recall@{k}":    round(np.mean(recalls),    4),
        f"f1@{k}":        round(np.mean(f1s),        4),
        f"ndcg@{k}":      round(np.mean(ndcgs),      4),
        f"hit_rate@{k}":  round(np.mean(hits),       4),
        "n_evaluated":    n_evaluated,
    }


def catalogue_coverage(all_recommendations: list[list[int]],
                        total_items: int) -> float:
    """Fraction of unique items ever recommended across all users."""
    recommended_items = {item for recs in all_recommendations for item in recs}
    return round(len(recommended_items) / total_items, 4)


# ── Display ───────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, model_name: str = "Model", k: int = 10) -> None:
    """Print a nicely formatted evaluation report."""
    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  EVALUATION @ K={k} — {model_name}")
    print(bar)
    for key, val in metrics.items():
        if key == "n_evaluated":
            print(f"  Users evaluated    : {val}")
        else:
            print(f"  {key:<22}: {val:.4f}")
    print(bar)


def print_comparison(results: dict[str, dict], k: int = 10) -> None:
    """Side-by-side table for multiple models."""
    bar = "=" * 72
    keys = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"]
    print(f"\n{bar}")
    print(f"  MODEL COMPARISON @ K={k}")
    print(bar)
    header = f"  {'Model':<28}" + "".join(f"{h:>10}" for h in keys)
    print(header)
    print("  " + "-" * 68)
    for name, m in results.items():
        row = f"  {name:<28}" + "".join(f"{m.get(h, 0):>10.4f}" for h in keys)
        print(row)
    print(bar)
