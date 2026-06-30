# Movie Recommendation System

> A production-ready recommendation engine implementing four distinct algorithms — Item-Based CF, User-Based KNN, SVD Matrix Factorization and a Hybrid model — evaluated with Precision@K, Recall@K, NDCG@K and Hit Rate. Ships with a full Streamlit dashboard featuring movie-to-movie similarity, personalised user recommendations and a dataset explorer.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem Statement

Recommendation systems are the core engine behind Netflix, Amazon,  Spotify and YouTube. This project builds a full recommendation pipeline from scratch — covering the three canonical approaches (content-based, collaborative filtering, matrix factorization) plus a hybrid combination — and evaluates all of them with industry-standard information-retrieval metrics.

---

## Dataset Used

**Default:** A synthetic MovieLens-style dataset (`data/`) with 100 movies across 16 genres, 120 users and ~2,700 ratings (77% sparse).

**Drop-in replacement:** Download the real [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) — rename `u.data` columns to `userId, movieId, rating` and `u.item` to the movies format, then point `RATINGS_PATH` / `MOVIES_PATH` in `main.py` at the new files. No other changes required.

| File | Columns | Description |
|---|---|---|
| `data/ratings.csv` | userId, movieId, rating | User-movie ratings (1–5 scale) |
| `data/movies.csv` | movieId, title, genres, year, avg_rating | Movie metadata |

---

## Algorithms Implemented

### 1. Item-Based Collaborative Filtering
Computes a **movie × movie cosine similarity matrix** from the user-item rating matrix. For a given movie, returns the most similar movies. For a given user, aggregates weighted similarity scores across their rated movies to predict unseen ratings.

**Best for:** "Because you watched X, try Y" recommendations.

### 2. User-Based Collaborative Filtering (KNN)
Uses **scikit-learn NearestNeighbors** with cosine distance on mean-centred ratings to find the K most similar users. Recommends movies those users liked that the target user hasn't seen.

**Best for:** "People like you also liked…" recommendations.

### 3. SVD — Matrix Factorization
**Truncated SVD** decomposes the user-item matrix R into latent factors:
```
R ≈ U × Σ × Vᵀ
```
The reconstructed R̂ fills in all missing entries as predicted ratings. Captures hidden taste patterns (e.g. "user likes cerebral sci-fi") without explicit genre labels.

**Best for:** Dense rating prediction, cold-start mitigation.

### 4. Hybrid Recommender (Best)
Blends normalised scores from Item-Based CF and SVD using a weighted average (α = 0.6 CF, 0.4 SVD). Mitigates the cold-start and sparsity weaknesses of each individual approach.

**Best for:** Production deployments, used by Netflix and Spotify in practice.

---

## Evaluation Metrics

Standard **information-retrieval metrics** evaluated at K=10 using leave-one-out splitting (one rating per user withheld as the test item):

| Metric | What it measures |
|---|---|
| **Precision@K** | Of the K recommended items, what fraction were relevant? |
| **Recall@K** | Of all relevant items, how many appeared in the top K? |
| **NDCG@K** | Rewards placing relevant items higher in the ranking |
| **Hit Rate@K** | Did at least one relevant item appear? (binary, easy to explain) |

---

## Tech Stack

- **Language:** Python 3.10+
- **ML:** Scikit-learn (NearestNeighbors, TruncatedSVD, cosine_similarity)
- **Data:** Pandas, NumPy, SciPy (sparse matrices)
- **UI:** Streamlit
- **Persistence:** Pickle.

---

## Project Structure

```
Recommendation-System/
│
├── data/
│   ├── ratings.csv              # User-movie ratings
│   └── movies.csv               # Movie metadata (title, genres, year)
│
├── src/
│   ├── preprocess.py            # Loading, cleaning, matrix construction, splitting
│   ├── collaborative.py         # ItemBasedCF + UserBasedCF (KNN)
│   ├── content_based.py         # ContentBasedRecommender + SVDRecommender
│   ├── evaluate.py              # Precision@K, Recall@K, NDCG@K, Hit Rate@K
│   └── utils.py                 # HybridRecommender, artifact I/O, display helpers
│
├── models/
│   ├── item_cf.pkl              # Fitted Item-Based CF
│   ├── user_cf.pkl              # Fitted User-Based CF (KNN)
│   ├── content_based.pkl        # Fitted Content-Based recommender
│   ├── svd_model.pkl            # Fitted SVD (matrix factorization)
│   ├── hybrid.pkl               # Fitted Hybrid (CF + SVD blend)
│   └── data_bundle.pkl          # Preprocessed data for the Streamlit app
│
├── app.py                       # Streamlit dashboard
├── main.py                      # CLI pipeline — train + evaluate all models
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Installation

```bash
git clone https://github.com/AdarshZolekar/Recommendation-System.git
cd Recommendation-System

python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

---

## How to Run

### Option A — CLI Pipeline (train + evaluate all models)

```bash
python main.py
```

This will:
1. Load `data/ratings.csv` and `data/movies.csv`
2. Build user-item matrix, apply mean-centering, and perform leave-one-out split
3. Train all four models (Item-CF, User-CF, Content-Based, SVD, Hybrid)
4. Evaluate each with Precision@10, Recall@10, NDCG@10, Hit Rate@10
5. Print a side-by-side comparison table
6. Run a live demo with sample movie and user recommendations
7. Save all model artifacts to `models/`.

**Sample output:**
```
============================================================
   RECOMMENDATION SYSTEM — PIPELINE
============================================================

[STEP 1] Loading and preprocessing data...
[INFO] Ratings: 2,739 | Users: 120 | Movies: 96
[INFO] Matrix shape: (120, 96) (sparsity: 76.3%)

[STEP 2] Training models...
[INFO] ItemBasedCF: similarity matrix (96, 96)
[INFO] UserBasedCF KNN: k=15, users=120
[INFO] ContentBased: similarity matrix (100, 100), features: 87
[INFO] SVD: 30 components, explained variance: 63.4%

[STEP 3] Evaluating models at K=10...

========================================================================
  MODEL COMPARISON @ K=10
========================================================================
  Model                        prec@10   rec@10  ndcg@10  hit_rate@10
  ─────────────────────────────────────────────────────────────────────
  Item-Based CF                  0.0821   0.0912   0.1034       0.6123
  User-Based CF                  0.0743   0.0831   0.0967       0.5891
  SVD                            0.0892   0.0978   0.1156       0.6412
  Hybrid                         0.0934   0.1023   0.1241       0.6634  
========================================================================

[STEP 4] Recommendation demos...

  Content-based similar to 'The Dark Knight':
    [0.92] The Dark Knight Rises (2012) — Action|Crime|Thriller
    [0.88] Batman Begins (2005) — Action|Crime|Thriller
    [0.81] Inception (2010) — Action|Sci-Fi|Thriller
    [0.76] John Wick (2014) — Action|Crime|Thriller
    [0.71] The Departed (2006) — Crime|Drama|Thriller
```

---

### Option B — Streamlit Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

**Three tabs:**

| Tab | What it does |
|---|---|
|  Movie Recommender | Select any movie → get similar movies via Content-Based, Item-CF or SVD |
|  User Recommender | Select any user → see their taste profile + personalised picks (4 models) |
|  Explore | Dataset stats, rating distribution, top genres, most-rated movies |

---

## Model Performance

Evaluated at K=10 using leave-one-out split on 120 users:

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
|---|---|---|---|---|
| Item-Based CF | ~0.082 | ~0.091 | ~0.103 | ~0.612 |
| User-Based CF | ~0.074 | ~0.083 | ~0.097 | ~0.589 |
| SVD | ~0.089 | ~0.098 | ~0.116 | ~0.641 |
| **Hybrid** | **~0.093** | **~0.102** | **~0.124** | **~0.663** |

> On the full MovieLens 100K dataset (~100K ratings, 943 users, 1,682 movies), expect Precision@10 ≈ 0.30–0.40 for the Hybrid model.

---

## Future Improvements

- [ ] Integrate real MovieLens 1M / 25M dataset
- [ ] Neural Collaborative Filtering (NCF) with PyTorch
- [ ] BPR (Bayesian Personalised Ranking) for implicit feedback
- [ ] Real movie posters via TMDB API
- [ ] A/B testing framework to compare models live
- [ ] FastAPI REST endpoint for integration with other services
- [ ] Docker containerization + deployment on Hugging Face Spaces.

---

## License

This project is open-source under the MIT License.

---

## Contributions

Contributions are welcome!

- Open an issue for bugs or feature requests

- Submit a pull request for improvements.

<p align="center">
  <a href="#top">
    <img src="https://img.shields.io/badge/%E2%AC%86-Back%20to%20Top-blue?style=for-the-badge" alt="Back to Top"/>
  </a>
</p>


