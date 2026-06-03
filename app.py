"""
app.py
------
Streamlit dashboard for the Movie Recommendation System.

Three modes:
  1. 🎬 Movie Recommender  — "Because you liked X, try Y…" (content-based)
  2. 👤 User Recommender   — Personalised picks for a user (hybrid CF + SVD)
  3. 🔍 Explore            — Browse movies, see stats, view rating distributions

Usage:
    streamlit run app.py
"""

import sys
import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import (
    load_pickle, enrich_recommendations, get_user_rated_movies,
    genre_badge_html, HybridRecommender
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main { background: #0f172a; }
  section[data-testid="stSidebar"] { background: #1e293b !important; }
  .movie-card {
    background: #1e293b;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    border-left: 4px solid #6366f1;
    transition: border-color 0.2s;
  }
  .movie-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
  }
  .movie-meta {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 0.4rem;
  }
  .score-bar-bg {
    background: #334155;
    border-radius: 20px;
    height: 6px;
    margin-top: 0.5rem;
  }
  .score-bar-fill {
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 20px;
    height: 6px;
  }
  .rank-badge {
    display: inline-block;
    background: #6366f1;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    line-height: 24px;
    text-align: center;
    margin-right: 8px;
  }
  .section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    padding: 0.5rem 0;
    border-bottom: 2px solid #334155;
    margin-bottom: 1rem;
  }
  .stSelectbox label, .stSlider label, .stRadio label {
    color: #cbd5e1 !important;
  }
  h1, h2, h3 { color: #f1f5f9 !important; }
  p, li { color: #cbd5e1; }
  .stMetric { background: #1e293b; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def load_all():
    data    = load_pickle("models/data_bundle.pkl")
    item_cf = load_pickle("models/item_cf.pkl")
    content = load_pickle("models/content_based.pkl")
    svd     = load_pickle("models/svd_model.pkl")
    hybrid  = HybridRecommender(item_cf=item_cf, svd=svd, alpha=0.6)
    return data, item_cf, content, svd, hybrid


try:
    data, item_cf, content, svd, hybrid = load_all()
    ratings = data["ratings"]
    movies  = data["movies"]
    matrix  = data["matrix"]
    ready   = True
except FileNotFoundError as e:
    ready   = False
    err_msg = str(e)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🎬 Movie Recommendation System")
st.markdown(
    "Powered by **Item-Based CF · User-Based KNN · SVD · Hybrid** models"
)

if not ready:
    st.error(f"⚠️ Models not found. Run `python main.py` first.\n\n{err_msg}")
    st.stop()

st.divider()

# All movie titles for search
all_titles  = movies.sort_values("title")["title"].tolist()
title_to_id = dict(zip(movies["title"], movies["movieId"]))
id_to_movie = movies.set_index("movieId")
all_users   = sorted(ratings["userId"].unique().tolist())


# ── Helper ────────────────────────────────────────────────────────────────────

def render_movie_card(rank: int, title: str, genres: str,
                      year, score: float, score_label: str = "Score") -> None:
    year_str = str(int(year)) if pd.notna(year) else "—"
    pct      = min(max(score, 0), 1) * 100
    genre_html = genre_badge_html(genres)
    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-title"><span class="rank-badge">{rank}</span>{title}</div>
      <div class="movie-meta">📅 {year_str}</div>
      <div style="margin-bottom:0.3rem">{genre_html}</div>
      <div style="font-size:0.78rem;color:#94a3b8">{score_label}: {score:.3f}</div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Navigation tabs ───────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🎬 Movie Recommender",
    "👤 User Recommender",
    "📊 Explore Dataset",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Movie-to-Movie (Content-Based)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p class="section-title">Find Movies Similar To…</p>',
                unsafe_allow_html=True)
    st.markdown("Uses **TF-IDF genre features** to find movies with similar themes and style.")

    col_input, col_opts = st.columns([3, 1])
    with col_input:
        selected_title = st.selectbox("🎥 Select a movie:", all_titles, key="cb_movie")
    with col_opts:
        n_recs_cb = st.slider("Top N", 3, 20, 8, key="cb_n")

    method = st.radio(
        "Similarity method:",
        ["Content-Based (Genres + Year)", "Item-Based CF (Rating Patterns)",
         "SVD Latent Factors"],
        horizontal=True,
    )

    if st.button("🔍 Find Similar Movies", key="cb_btn"):
        movie_id = title_to_id.get(selected_title)
        if movie_id is None:
            st.warning("Movie not found.")
        else:
            # Show selected movie info
            m_info = id_to_movie.loc[movie_id] if movie_id in id_to_movie.index else None
            if m_info is not None:
                st.markdown(f"**Selected:** {genre_badge_html(m_info.get('genres',''))}",
                            unsafe_allow_html=True)

            # Get recs based on method
            if method.startswith("Content"):
                recs = content.get_similar_movies(movie_id, top_n=n_recs_cb)
                score_col = "similarity_score"
            elif method.startswith("Item"):
                raw  = item_cf.get_similar_movies(movie_id, top_n=n_recs_cb)
                recs = raw.merge(movies[["movieId","title","genres","year"]],
                                 on="movieId", how="left")
                score_col = "similarity_score"
            else:
                raw  = svd.get_similar_movies(movie_id, top_n=n_recs_cb)
                recs = raw.merge(movies[["movieId","title","genres","year"]],
                                 on="movieId", how="left")
                score_col = "similarity_score"

            if recs.empty:
                st.info("Not enough data to generate recommendations for this movie.")
            else:
                st.markdown(f"<br>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    render_movie_card(
                        i, row["title"], row.get("genres",""),
                        row.get("year", ""), row[score_col],
                        score_label="Similarity"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Personalised User Recommendations
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<p class="section-title">Personalised Recommendations</p>',
                unsafe_allow_html=True)
    st.markdown("Select a user to see their taste profile and personalised picks.")

    ucol1, ucol2, ucol3 = st.columns([2, 1, 1])
    with ucol1:
        selected_uid = st.selectbox("👤 Select User ID:", all_users, key="uid_sel")
    with ucol2:
        n_recs_u = st.slider("Top N", 3, 20, 8, key="u_n")
    with ucol3:
        rec_method = st.selectbox("Model:", [
            "Hybrid (CF + SVD)", "Item-Based CF", "User-Based CF", "SVD"
        ], key="u_method")

    if st.button("🎯 Get Recommendations", key="u_btn"):

        # Show user's taste profile
        liked = get_user_rated_movies(selected_uid, ratings, movies, min_rating=4.0)
        all_rated = get_user_rated_movies(selected_uid, ratings, movies)
        avg_rating = all_rated["rating"].mean() if not all_rated.empty else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Movies Rated", len(all_rated))
        m2.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
        m3.metric("Top Genres", ", ".join(
            pd.Series("|".join(liked["genres"].dropna()).split("|"))
            .value_counts().head(2).index.tolist()
        ) if not liked.empty else "—")

        if not liked.empty:
            st.markdown("**Recently liked:**")
            liked_tags = " · ".join(
                f"*{r['title']}* ({r['rating']:.0f}★)"
                for _, r in liked.head(4).iterrows()
            )
            st.markdown(liked_tags)

        st.divider()
        st.markdown(f"**Top {n_recs_u} picks using {rec_method}:**")

        # Generate recs
        if rec_method == "Hybrid (CF + SVD)":
            raw_recs = hybrid.recommend_for_user(selected_uid, top_n=n_recs_u)
            recs = enrich_recommendations(raw_recs, movies, score_col="hybrid_score")
            score_col = "hybrid_score"
        elif rec_method == "Item-Based CF":
            raw_recs = item_cf.recommend_for_user(selected_uid, top_n=n_recs_u)
            recs = enrich_recommendations(raw_recs, movies, score_col="predicted_rating")
            score_col = "predicted_rating"
        elif rec_method == "User-Based CF":
            raw_recs = data["user_cf"].recommend_for_user(selected_uid, top_n=n_recs_u) \
                       if "user_cf" in data else pd.DataFrame()
            if raw_recs is None or raw_recs.empty:
                # Fallback: use user_cf from loaded model
                try:
                    user_cf_model = load_pickle("models/user_cf.pkl")
                    raw_recs = user_cf_model.recommend_for_user(selected_uid, top_n=n_recs_u)
                except Exception:
                    raw_recs = pd.DataFrame()
            recs = enrich_recommendations(raw_recs, movies, score_col="predicted_rating") \
                   if not raw_recs.empty else pd.DataFrame()
            score_col = "predicted_rating"
        else:  # SVD
            raw_recs = svd.recommend_for_user(selected_uid, top_n=n_recs_u)
            recs = enrich_recommendations(raw_recs, movies, score_col="predicted_rating")
            score_col = "predicted_rating"

        if recs is None or recs.empty:
            st.info("Not enough rating history to generate recommendations for this user.")
        else:
            # Normalise score to [0,1] for the progress bar
            scores = recs[score_col].values.astype(float)
            mn, mx = scores.min(), scores.max()
            norm_scores = (scores - mn) / (mx - mn + 1e-9)

            for i, ((_, row), ns) in enumerate(zip(recs.iterrows(), norm_scores), 1):
                render_movie_card(
                    i, row["title"], row.get("genres",""),
                    row.get("year",""), ns,
                    score_label=score_col.replace("_"," ").title()
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Explore Dataset
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<p class="section-title">Dataset Explorer</p>',
                unsafe_allow_html=True)

    # Top-level stats
    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Total Movies",  f"{len(movies):,}")
    ec2.metric("Total Users",   f"{ratings['userId'].nunique():,}")
    ec3.metric("Total Ratings", f"{len(ratings):,}")
    ec4.metric("Avg Rating",    f"{ratings['rating'].mean():.2f} ⭐")

    st.markdown("---")
    ecol1, ecol2 = st.columns(2)

    # Rating distribution
    with ecol1:
        st.markdown("**Rating Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#1e293b")
        bins = [0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25]
        counts, _, patches = ax.hist(ratings["rating"], bins=bins,
                                      color="#6366f1", edgecolor="#334155", rwidth=0.8)
        ax.set_xlabel("Rating", color="#94a3b8")
        ax.set_ylabel("Count",  color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["left","bottom"]:
            ax.spines[spine].set_color("#334155")
        st.pyplot(fig)
        plt.close(fig)

    # Top genres
    with ecol2:
        st.markdown("**Top Genres by Movie Count**")
        all_genres = pd.Series(
            "|".join(movies["genres"].dropna()).split("|")
        ).value_counts().head(12)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig2.patch.set_facecolor("#1e293b")
        ax2.set_facecolor("#1e293b")
        colors = ["#6366f1","#8b5cf6","#a78bfa","#c4b5fd",
                  "#7c3aed","#4f46e5","#3730a3","#312e81",
                  "#06b6d4","#0ea5e9","#3b82f6","#60a5fa"]
        ax2.barh(all_genres.index[::-1], all_genres.values[::-1],
                 color=colors[:len(all_genres)], edgecolor="#334155")
        ax2.tick_params(colors="#94a3b8")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ["left","bottom"]:
            ax2.spines[spine].set_color("#334155")
        st.pyplot(fig2)
        plt.close(fig2)

    # Most rated movies
    st.markdown("**Most Rated Movies**")
    most_rated = (
        ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .merge(movies[["movieId","title","genres","year"]], on="movieId")
        .rename(columns={"count":"num_ratings","mean":"avg_rating"})
        .sort_values("num_ratings", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    most_rated["avg_rating"] = most_rated["avg_rating"].round(2)
    most_rated.index += 1
    st.dataframe(
        most_rated[["title","genres","year","num_ratings","avg_rating"]],
        use_container_width=True,
        height=300,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Movie Recommendation System · Item-Based CF · User-Based KNN · SVD · Hybrid · "
    "Built with Scikit-learn & Streamlit"
)
