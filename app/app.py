import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import torch
import pandas as pd
import numpy as np
from model.lightgcn import LightGCN
from torch_geometric.data import Data
import torch.serialization
import urllib.parse

# ======================================
# Page Config
# ======================================
st.set_page_config(
    page_title="CineGraph — GNN Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# CSS Styling
# ======================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 52px; font-weight: 900;
    color: #0f172a; letter-spacing: -1px; line-height: 1.1;
}
.subtitle { font-size: 15px; color: #64748b; font-weight: 300; margin-top: 4px; }

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 22px; font-weight: 700; color: #0f172a;
    margin-bottom: 14px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px;
}

.stat-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 20px; text-align: center;
}
.stat-value { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 900; color: #0f172a; }
.stat-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

.movie-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.movie-rank { font-family:'Playfair Display',serif; font-size:26px; font-weight:900; color:#cbd5e1; margin-right:12px; }
.movie-title { font-size: 16px; font-weight: 600; color: #0f172a; }
.movie-year  { font-size: 13px; color: #94a3b8; margin-left: 6px; }

.badge       { display:inline-block; border-radius:20px; padding:3px 10px; font-size:11px; font-weight:500; letter-spacing:0.4px; text-transform:uppercase; margin-right:5px; margin-bottom:3px; }
.badge-genre { background:#f0fdf4; color:#15803d; }
.badge-rating{ background:#fefce8; color:#a16207; }
.badge-pop   { background:#eff6ff; color:#1d4ed8; }

.score-bar-bg { background:#f1f5f9; border-radius:100px; height:5px; width:100%; margin-top:7px; }
.score-bar    { background:linear-gradient(90deg,#3b82f6,#06b6d4); height:5px; border-radius:100px; }

.poster-card {
    background:white; border:1px solid #e2e8f0; border-radius:14px;
    overflow:hidden; text-align:center; padding-bottom:10px;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
.poster-title { font-size:13px; font-weight:600; color:#0f172a; padding:8px 10px 2px; line-height:1.3; }
.poster-meta  { font-size:11px; color:#94a3b8; padding:0 8px 4px; }

.trend-card {
    background:white; border:1px solid #e2e8f0; border-radius:12px;
    padding:12px 16px; margin-bottom:8px;
}
.search-result {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
    padding:14px 18px; margin-bottom:8px;
}

.stButton > button {
    background: linear-gradient(135deg,#1e40af,#0ea5e9) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    padding:10px 22px !important; font-weight:500 !important; font-size:15px !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stSidebarContent"] { background:#0f172a; }
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] p { color:#94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ======================================
# Genre Metadata Loader
# ======================================
@st.cache_data
def load_genre_data():
    genre_names = ["unknown","Action","Adventure","Animation","Children's",
                   "Comedy","Crime","Documentary","Drama","Fantasy",
                   "Film-Noir","Horror","Musical","Mystery","Romance",
                   "Sci-Fi","Thriller","War","Western"]
    item_file = os.path.join(ROOT_DIR, "ml-100k", "u.item")
    movie_genres, movie_years = {}, {}
    try:
        with open(item_file, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 5: continue
                mid = int(parts[0])
                title_raw = parts[1]
                year = title_raw[title_raw.rfind("(")+1:title_raw.rfind(")")] if "(" in title_raw else ""
                flags = parts[5:24]
                genres = [genre_names[i] for i,g in enumerate(flags) if g=="1" and genre_names[i]!="unknown"]
                movie_genres[mid] = genres if genres else ["Unknown"]
                movie_years[mid] = year
    except Exception:
        pass
    return movie_genres, movie_years


# ======================================
# Poster URL helper
# ======================================
POSTER_COLORS = [
    "1a1a2e/e94560","16213e/0f3460","0f3460/e94560",
    "1b262c/415a77","2c3e50/e74c3c","1a1a1a/f39c12",
    "2d3561/c05c7e","1f4068/1b262c","162447/1f4068",
    "0d0d0d/f5a623","2e4057/048a81","3d5a80/e0fbfc",
    "22223b/4a4e69","3a0ca3/f72585","560bad/480ca8"
]

def get_poster_url(movie_id, title):
    color = POSTER_COLORS[movie_id % len(POSTER_COLORS)]
    bg, fg = color.split("/")
    short  = title.split("(")[0].strip()[:18]
    label  = urllib.parse.quote(short)
    return f"https://placehold.co/140x200/{bg}/{fg}?text={label}&font=playfair-display"


# ======================================
# Load Everything (cached)
# ======================================
@st.cache_resource
def load_all():
    torch.serialization.add_safe_globals([Data])
    DATA_PATH = os.path.join(ROOT_DIR, "data", "ratings_with_titles.csv")
    df = pd.read_csv(DATA_PATH)
    movie_map     = df.set_index("item")["title"].to_dict()
    avg_ratings   = df.groupby("item")["rating"].mean().to_dict()
    rating_counts = df.groupby("item")["rating"].count().to_dict()
    user_history  = df.groupby("user")["item"].apply(list).to_dict()

    GRAPH_PATH = os.path.join(ROOT_DIR, "graph", "graph_data.pt")
    pkg        = torch.load(GRAPH_PATH, weights_only=False)
    num_users  = pkg["num_users"]
    num_items  = pkg["num_items"]

    MODEL_PATH = os.path.join(ROOT_DIR, "model", "lightgcn_model.pt")
    model = LightGCN(num_users, num_items)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return df, movie_map, avg_ratings, rating_counts, user_history, num_users, num_items, model

df, movie_map, avg_ratings, rating_counts, user_history, num_users, num_items, model = load_all()
movie_genres, movie_years = load_genre_data()

all_genres  = sorted({g for gs in movie_genres.values() for g in gs} - {"Unknown","unknown"})
all_titles  = sorted(movie_map.values())
title_to_id = {v: k for k, v in movie_map.items()}


# ======================================
# Core Logic Functions
# ======================================
def get_recommendations(user_id, top_k=10, exclude_seen=True, genre_filter=None, min_ratings=5):
    with torch.no_grad():
        user_emb = model.user_emb.weight[user_id]
        scores   = torch.matmul(user_emb, model.item_emb.weight.T).numpy()
    seen = set(user_history.get(user_id+1, []))
    results = []
    for idx in range(len(scores)):
        mid = idx+1
        if exclude_seen and mid in seen: continue
        if rating_counts.get(mid,0) < min_ratings: continue
        if genre_filter and genre_filter != "All":
            if genre_filter not in movie_genres.get(mid,[]): continue
        results.append((mid, movie_map.get(mid,"Unknown"), float(scores[idx])))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]


def get_similar_movies(query_title, top_n=8):
    mid = title_to_id.get(query_title)
    if mid is None: return []
    idx = mid - 1
    with torch.no_grad():
        all_item_emb = model.item_emb.weight
        target = all_item_emb[idx]
        sims   = torch.nn.functional.cosine_similarity(target.unsqueeze(0), all_item_emb)
        sims[idx] = -1
        top    = torch.topk(sims, top_n).indices.tolist()
    return [(i+1, movie_map.get(i+1,"Unknown"), float(sims[i].item())) for i in top]


def find_similar_users(user_id, top_n=5):
    with torch.no_grad():
        embs = model.user_emb.weight
        sims = torch.nn.functional.cosine_similarity(embs[user_id].unsqueeze(0), embs)
        sims[user_id] = -1
        return torch.topk(sims, top_n).indices.tolist()


def get_trending(top_n=10, mode="popular"):
    stats = [(mid, title, avg_ratings.get(mid,0), rating_counts.get(mid,0))
             for mid,title in movie_map.items() if rating_counts.get(mid,0) >= 20]
    stats.sort(key=lambda x: x[3] if mode=="popular" else x[2], reverse=True)
    return stats[:top_n]


# ======================================
# Sidebar
# ======================================
with st.sidebar:
    st.markdown("## 🎬 CineGraph")
    st.markdown('<p style="color:#475569;font-size:13px;">GNN-Powered Movie Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">User Selection</p>', unsafe_allow_html=True)
    user_id = st.selectbox("User ID", options=list(range(num_users)), format_func=lambda x: f"User #{x+1}")
    st.markdown("---")
    st.markdown('<p style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Recommendation Filters</p>', unsafe_allow_html=True)
    top_k        = st.slider("No. of Recommendations", 5, 20, 10)
    genre_filter = st.selectbox("Genre Filter", ["All"] + all_genres)
    exclude_seen = st.checkbox("Exclude already-seen movies", value=True)
    min_ratings  = st.slider("Min. community ratings", 1, 50, 5)
    st.markdown("---")
    st.markdown('<p style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Display Options</p>', unsafe_allow_html=True)
    show_posters       = st.checkbox("Show movie posters", value=True)
    show_score_bars    = st.checkbox("Show relevance bars", value=True)
    show_user_history  = st.checkbox("Show user history tab", value=True)
    show_similar_users = st.checkbox("Show similar users", value=False)


# ======================================
# Header
# ======================================
st.markdown('<div class="main-title">CineGraph</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Graph Neural Network · LightGCN · MovieLens 100K · Personalized Movie Intelligence</div>', unsafe_allow_html=True)
st.markdown("")

# ======================================
# TABS
# ======================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "🔍 Movie Search", "🔥 Trending", "📼 My History"])


# ══════════════════════════════════════
# TAB 1 — Personalized Recommendations
# ══════════════════════════════════════
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    seen_count = len(set(user_history.get(user_id+1,[])))
    for col, val, lbl in [(c1, f"{num_users:,}", "Users"), (c2, f"{num_items:,}", "Movies"),
                           (c3, str(seen_count), "Rated by You"), (c4, str(top_k), "Recs")]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    btn_col, _ = st.columns([1,3])
    with btn_col:
        gen_btn = st.button("✨ Generate Recommendations", use_container_width=True)

    if gen_btn:
        with st.spinner("Running GNN inference..."):
            recs = get_recommendations(user_id, top_k, exclude_seen,
                                        genre_filter if genre_filter!="All" else None, min_ratings)
        if not recs:
            st.warning("No results with current filters. Try relaxing genre or min-ratings.")
        else:
            scores = [r[2] for r in recs]
            max_s, min_s = max(scores), min(scores)
            rng = max_s - min_s if max_s != min_s else 1

            st.markdown(f'<div class="section-header">🎯 Top {len(recs)} Picks for User #{user_id+1}</div>', unsafe_allow_html=True)

            # Poster grid (top 5)
            if show_posters:
                st.markdown("**🎞️ Top Picks at a Glance**")
                pcols = st.columns(min(5, len(recs)))
                for ci, (mid, title, score) in enumerate(recs[:5]):
                    with pcols[ci]:
                        avg_r = avg_ratings.get(mid, 0)
                        year  = movie_years.get(mid,"")
                        purl  = get_poster_url(mid, title)
                        st.markdown(f"""
                        <div class="poster-card">
                            <img src="{purl}" style="width:100%;border-radius:8px 8px 0 0;" onerror="this.src='https://placehold.co/140x200/1e293b/94a3b8?text=🎬'"/>
                            <div class="poster-title">{title.split('(')[0].strip()}</div>
                            <div class="poster-meta">⭐ {avg_r:.1f} &nbsp;·&nbsp; {year}</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown("")

            # Full ranked list
            st.markdown("**📋 Full Ranked List**")
            for i, (mid, title, score) in enumerate(recs, 1):
                genres = movie_genres.get(mid, [])
                year   = movie_years.get(mid,"")
                avg_r  = avg_ratings.get(mid, None)
                cnt    = rating_counts.get(mid, 0)
                norm   = (score - min_s) / rng
                g_html = "".join([f'<span class="badge badge-genre">{g}</span>' for g in genres[:3]])
                r_html = f'<span class="badge badge-rating">⭐ {avg_r:.1f} ({cnt})</span>' if avg_r else ""
                y_html = f'<span class="movie-year">({year})</span>' if year else ""
                b_html = f'<div class="score-bar-bg"><div class="score-bar" style="width:{int(norm*100)}%"></div></div><span style="font-size:11px;color:#94a3b8;">GNN Score: {score:.4f}</span>' if show_score_bars else ""
                st.markdown(f"""
                <div class="movie-card">
                    <span class="movie-rank">#{i:02d}</span>
                    <span class="movie-title">{title}</span>{y_html}
                    <div style="margin-top:7px;">{g_html}{r_html}</div>
                    {b_html}
                </div>""", unsafe_allow_html=True)

            # ⬇️ Download
            st.markdown("")
            rec_df = pd.DataFrame(
                [(mid, title, f"{s:.4f}", f"{avg_ratings.get(mid,0):.2f}",
                  rating_counts.get(mid,0), ", ".join(movie_genres.get(mid,[])), movie_years.get(mid,""))
                 for mid,title,s in recs],
                columns=["Movie ID","Title","GNN Score","Avg Rating","# Ratings","Genres","Year"]
            )
            st.download_button("⬇️ Download Recommendations as CSV",
                                rec_df.to_csv(index=False).encode("utf-8"),
                                f"cinegraph_recs_user{user_id+1}.csv", "text/csv",
                                use_container_width=True)


# ══════════════════════════════════════
# TAB 2 — Movie Search + Similar Movies
# ══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 Movie Search & Discovery</div>', unsafe_allow_html=True)
    st.markdown("Type any movie name to instantly find **similar titles** powered by GNN item embeddings.")
    st.markdown("")

    search_query = st.text_input("🎬 Search movie...", placeholder="e.g. Toy Story, Star Wars, Titanic, Fargo")

    if search_query:
        matches = [t for t in all_titles if search_query.lower() in t.lower()]
        if not matches:
            st.warning(f"No movies found for **'{search_query}'**. Try a shorter keyword.")
        else:
            selected = st.selectbox("Select exact movie:", matches[:20])
            if selected:
                mid_sel  = title_to_id.get(selected)
                genres_s = movie_genres.get(mid_sel, [])
                year_s   = movie_years.get(mid_sel, "")
                avg_s    = avg_ratings.get(mid_sel, 0)
                cnt_s    = rating_counts.get(mid_sel, 0)

                info_c, sim_c = st.columns([1,2], gap="large")

                with info_c:
                    st.markdown("**Selected Movie**")
                    purl = get_poster_url(mid_sel, selected)
                    st.markdown(f"""
                    <div style="background:white;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;text-align:center;padding-bottom:14px;">
                        <img src="{purl}" style="width:100%;border-radius:8px 8px 0 0;" onerror="this.src='https://placehold.co/200x280/1e293b/94a3b8?text=🎬'"/>
                        <div style="font-family:'Playfair Display',serif;font-size:16px;font-weight:700;padding:10px 10px 4px;color:#0f172a;">{selected.split("(")[0].strip()}</div>
                        <div style="font-size:12px;color:#94a3b8;padding:0 8px 6px;">{year_s}</div>
                        <div style="padding:4px 10px;">{"".join([f'<span class="badge badge-genre">{g}</span>' for g in genres_s[:3]])}</div>
                        <div style="font-size:13px;color:#64748b;margin-top:6px;">⭐ {avg_s:.2f} &nbsp;·&nbsp; {cnt_s} ratings</div>
                    </div>""", unsafe_allow_html=True)

                with sim_c:
                    st.markdown(f"**Movies Similar to** *{selected.split('(')[0].strip()}*")
                    with st.spinner("Finding similar movies..."):
                        similars = get_similar_movies(selected, top_n=8)

                    if show_posters:
                        cols_row1 = st.columns(4)
                        cols_row2 = st.columns(4)
                        for ci, (smid, stitle, sscore) in enumerate(similars[:8]):
                            savg  = avg_ratings.get(smid, 0)
                            syear = movie_years.get(smid,"")
                            spurl = get_poster_url(smid, stitle)
                            col   = cols_row1[ci % 4] if ci < 4 else cols_row2[ci % 4]
                            with col:
                                st.markdown(f"""
                                <div class="poster-card" style="margin-bottom:10px;">
                                    <img src="{spurl}" style="width:100%;border-radius:8px 8px 0 0;" onerror="this.src='https://placehold.co/140x200/1e293b/94a3b8?text=🎬'"/>
                                    <div class="poster-title">{stitle.split("(")[0].strip()}</div>
                                    <div class="poster-meta">⭐ {savg:.1f} · {syear}</div>
                                </div>""", unsafe_allow_html=True)
                    else:
                        for i, (smid, stitle, sscore) in enumerate(similars, 1):
                            savg = avg_ratings.get(smid, 0)
                            syear= movie_years.get(smid,"")
                            sg   = "".join([f'<span class="badge badge-genre">{g}</span>' for g in movie_genres.get(smid,[])[:2]])
                            st.markdown(f"""
                            <div class="search-result">
                                <span class="movie-rank" style="font-size:18px;">#{i}</span>
                                <span class="movie-title">{stitle}</span><span class="movie-year">({syear})</span>
                                <div style="margin-top:5px;">{sg}<span class="badge badge-rating">⭐ {savg:.1f}</span></div>
                            </div>""", unsafe_allow_html=True)

                    sim_df = pd.DataFrame(
                        [(m, t, f"{s:.4f}", f"{avg_ratings.get(m,0):.2f}", movie_years.get(m,""))
                         for m,t,s in similars],
                        columns=["Movie ID","Title","Similarity","Avg Rating","Year"]
                    )
                    st.download_button("⬇️ Download Similar Movies CSV",
                                        sim_df.to_csv(index=False).encode("utf-8"),
                                        f"similar_to_{selected[:20].replace(' ','_')}.csv","text/csv")


# ══════════════════════════════════════
# TAB 3 — Trending Movies
# ══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🔥 Trending & Top Rated</div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2, gap="large")

    def render_trend_list(movies, badge_fn):
        if show_posters:
            pg1, pg2 = st.columns(2)
            for ci, row in enumerate(movies[:6]):
                mid, title, avg_r, cnt = row
                purl = get_poster_url(mid, title)
                year = movie_years.get(mid,"")
                col  = pg1 if ci % 2 == 0 else pg2
                with col:
                    st.markdown(f"""
                    <div class="poster-card" style="margin-bottom:10px;">
                        <img src="{purl}" style="width:100%;border-radius:8px 8px 0 0;" onerror="this.src='https://placehold.co/140x200/1e293b/94a3b8?text=🎬'"/>
                        <div class="poster-title">{title.split("(")[0].strip()}</div>
                        <div class="poster-meta">{badge_fn(avg_r, cnt)} · {year}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            for i, (mid, title, avg_r, cnt) in enumerate(movies, 1):
                year = movie_years.get(mid,"")
                genres = "".join([f'<span class="badge badge-genre">{g}</span>' for g in movie_genres.get(mid,[])[:2]])
                st.markdown(f"""
                <div class="trend-card">
                    <span class="movie-rank" style="font-size:20px;min-width:32px;">#{i}</span>
                    <div>
                        <span class="movie-title">{title}</span><span class="movie-year">({year})</span>
                        <div style="margin-top:4px;">{genres}{badge_fn(avg_r, cnt)}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with t_col1:
        st.markdown("### 📈 Most Popular")
        st.caption("Ranked by number of community ratings")
        popular = get_trending(10, "popular")
        render_trend_list(popular, lambda a,c: f'<span class="badge badge-pop">👥 {c} ratings</span>')

    with t_col2:
        st.markdown("### ⭐ Highest Rated")
        st.caption("Ranked by average community rating (min 20 ratings)")
        top_rated = get_trending(10, "rated")
        render_trend_list(top_rated, lambda a,c: f'<span class="badge badge-rating">⭐ {a:.2f}</span>')

    st.markdown("")
    trend_df = pd.DataFrame(
        [(mid, title, f"{avg_r:.2f}", cnt, ", ".join(movie_genres.get(mid,[])), movie_years.get(mid,""))
         for mid,title,avg_r,cnt in popular],
        columns=["Movie ID","Title","Avg Rating","# Ratings","Genres","Year"]
    )
    st.download_button("⬇️ Download Trending Movies CSV",
                        trend_df.to_csv(index=False).encode("utf-8"),
                        "cinegraph_trending.csv","text/csv")


# ══════════════════════════════════════
# TAB 4 — User History & Profile
# ══════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-header">📼 User #{user_id+1} — Watch Profile</div>', unsafe_allow_html=True)
    history_ids = user_history.get(user_id+1, [])
    hist_col, prof_col = st.columns([3,2], gap="large")

    with hist_col:
        st.markdown(f"**Rated Movies ({len(history_ids)} total)**")
        if history_ids:
            hist_rows = [{"Title": movie_map.get(m,"Unknown"), "Year": movie_years.get(m,""),
                          "Genres": ", ".join(movie_genres.get(m,[])[:2]),
                          "Avg ⭐": f"{avg_ratings.get(m,0):.1f}"} for m in history_ids]
            hist_df = pd.DataFrame(hist_rows)
            st.dataframe(hist_df, use_container_width=True, hide_index=True, height=400)
            st.download_button("⬇️ Download My History CSV",
                                hist_df.to_csv(index=False).encode("utf-8"),
                                f"history_user{user_id+1}.csv","text/csv")
        else:
            st.info("No rating history found for this user.")

    with prof_col:
        if history_ids:
            st.markdown("**Genre Breakdown**")
            genre_count = {}
            for mid in history_ids:
                for g in movie_genres.get(mid,[]):
                    genre_count[g] = genre_count.get(g,0)+1
            gc_df = pd.DataFrame(list(genre_count.items()), columns=["Genre","Count"])
            st.bar_chart(gc_df.sort_values("Count",ascending=False).head(8).set_index("Genre"))

        if show_similar_users:
            st.markdown("**👥 Similar Users**")
            for uid in find_similar_users(user_id, 5):
                cnt = len(user_history.get(uid+1,[]))
                st.markdown(f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px 16px;margin-bottom:7px;">
                    <b style="color:#0f172a;">User #{uid+1}</b>
                    <span style="color:#94a3b8;font-size:13px;margin-left:8px;">{cnt} movies rated</span>
                </div>""", unsafe_allow_html=True)

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#cbd5e1;font-size:13px;padding:6px 0;">
    CineGraph &nbsp;·&nbsp; LightGCN on MovieLens 100K &nbsp;·&nbsp; Streamlit + PyTorch Geometric
</div>""", unsafe_allow_html=True)
