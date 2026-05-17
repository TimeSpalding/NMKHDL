"""
demo.py — Streamlit demo: Hệ Khuyến Nghị Phim
Chạy: streamlit run demo.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import RATINGS_PATH, MOVIES_PATH, TAGS_PATH, LINKS_PATH, RANDOM_STATE, MIN_USER_RATINGS
from models import ItemItemCF, UserUserCF, SVDRecommender, ContentBasedCF, HybridRecommender

# Cấu hình trang 
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #7c6af7;
    }
    .metric-box .label { color: #aaa; font-size: 13px; }
    .metric-box .value { color: #fff; font-size: 22px; font-weight: 700; }
    .rec-card {
        background: #16213e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .badge {
        background: #7c6af7;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }
    h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# Load & cache dữ liệu 
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data(sample_size: int):
    ratings_full = pd.read_csv(RATINGS_PATH)
    movies       = pd.read_csv(MOVIES_PATH)
    try:
        tags = pd.read_csv(TAGS_PATH)
    except Exception:
        tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    ratings = ratings_full.sample(n=min(sample_size, len(ratings_full)),
                                   random_state=RANDOM_STATE).reset_index(drop=True)
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # Lọc cold-start users
    ucnt = ratings.groupby("userId")["rating"].count()
    valid = ucnt[ucnt >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["userId"].isin(valid)].reset_index(drop=True)

    train = ratings.sample(frac=0.8, random_state=RANDOM_STATE)
    test  = ratings.drop(train.index)
    train_matrix = train.pivot_table(index="userId", columns="movieId", values="rating")

    return ratings, movies, tags, train, test, train_matrix


@st.cache_resource(show_spinner="Đang huấn luyện mô hình (lần đầu ~30-60 giây)...")
def train_models(_train_matrix, _movies, _tags):
    item_cf = ItemItemCF(_train_matrix)
    user_cf = UserUserCF(_train_matrix)
    svd     = SVDRecommender(_train_matrix)
    cb      = ContentBasedCF(_movies, _train_matrix, tags_df=_tags)
    hybrid  = HybridRecommender(item_cf, cb, alpha=0.7)
    return item_cf, user_cf, svd, cb, hybrid


#  Sidebar 
with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.caption("Đồ án Nhập Môn KHDL — MovieLens")
    st.divider()

    sample_size = st.slider("Số ratings lấy mẫu", 1000, 10000, 5000, 500)
    n_recs = st.slider("Số phim gợi ý hiển thị", 5, 20, 10)
    st.divider()

    tab_choice = st.radio("Chọn tính năng", [
        " Gợi ý cho User",
        " So sánh 3 mô hình",
        "Tìm phim tương tự",
        "User Mới (Cold Start)",
    ])

# Load data & models 
try:
    ratings, movies, tags, train, test, train_matrix = load_data(sample_size)
    item_cf, user_cf, svd, cb, hybrid = train_models(train_matrix, movies, tags)
except FileNotFoundError:
    st.error("Không tìm thấy file CSV. Hãy đặt `ratings.csv` và `movies.csv` vào thư mục `data/`.")
    st.code("data/\n  ratings.csv\n  movies.csv\n  tags.csv\n  links.csv")
    st.stop()

all_users  = sorted(train_matrix.index.tolist())
all_movies = movies[movies["movieId"].isin(train_matrix.columns)].sort_values("title")



# TAB 1 — Gợi ý cho User

if tab_choice == " Gợi ý cho User":
    st.title("Gợi ý phim cho User")

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        user_id = st.selectbox("Chọn User ID", all_users)
        model_choice = st.radio("Dùng mô hình", [
            "Item-Item CF", "User-User CF", "SVD",
            "Content-Based", "Hybrid (0.7 CF + 0.3 CB)",
        ])

    with col_info:
        user_ratings = train_matrix.loc[user_id].dropna()
        st.markdown(f"**User {user_id}** đã đánh giá **{len(user_ratings)} phim** trong tập train")

        # Hiện lịch sử đánh giá của user
        if st.checkbox("Xem lịch sử đánh giá"):
            history = (
                user_ratings.reset_index()
                .rename(columns={"movieId": "movieId", user_id: "rating"})
                .merge(movies[["movieId", "title", "genres"]], on="movieId")
                .sort_values("rating", ascending=False)
                .head(10)
            )
            history.columns = ["movieId", "Rating ⭐", "Tên phim", "Thể loại"]
            st.dataframe(history[["Tên phim", "Rating ⭐", "Thể loại"]], use_container_width=True)

    st.divider()

    # Lấy gợi ý
    model_map = {
        "Item-Item CF":            item_cf,
        "User-User CF":            user_cf,
        "SVD":                     svd,
        "Content-Based":           cb,
        "Hybrid (0.7 CF + 0.3 CB)": hybrid,
    }
    model = model_map[model_choice]
    recs  = model.recommend(user_id, n=n_recs)

    if not recs:
        st.warning("Không có gợi ý cho user này.")
    else:
        rec_df = (
            pd.DataFrame(recs, columns=["movieId", "pred_rating"])
            .merge(movies[["movieId", "title", "genres"]], on="movieId")
        )

        # Biểu đồ bar ngang
        fig, ax = plt.subplots(figsize=(9, max(4, len(rec_df) * 0.45)))
        colors = plt.cm.RdYlGn(np.linspace(0.4, 0.85, len(rec_df)))
        short  = rec_df["title"].str[:40]
        bars   = ax.barh(range(len(rec_df)), rec_df["pred_rating"], color=colors)
        ax.set_yticks(range(len(rec_df)))
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Predicted Rating ⭐")
        ax.set_title(f"Top-{n_recs} Gợi Ý ({model_choice}) cho User {user_id}",
                     fontsize=12, fontweight="bold")
        for bar, val in zip(bars, rec_df["pred_rating"]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 5.8)
        ax.axvline(4.0, color="gray", ls="--", alpha=0.4, label="4★ threshold")
        plt.tight_layout()
        st.pyplot(fig)

        # Bảng chi tiết
        st.subheader("Chi tiết gợi ý")
        rec_df["rank"] = range(1, len(rec_df) + 1)
        rec_df["pred_rating"] = rec_df["pred_rating"].round(2)
        st.dataframe(
            rec_df[["rank", "title", "genres", "pred_rating"]]
            .rename(columns={"rank": "#", "title": "Phim", "genres": "Thể loại", "pred_rating": "⭐ Dự đoán"}),
            use_container_width=True,
            hide_index=True,
        )


# TAB 2 — So sánh 3 mô hình

elif tab_choice == "⚖️ So sánh 3 mô hình":
    st.title("So sánh 5 mô hình: CF · SVD · Content-Based · Hybrid")

    @st.cache_data(show_spinner="Đang tính MAE / RMSE...")
    def compute_metrics(_item_cf, _user_cf, _svd, _cb, _hybrid, _test):
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        results = {}
        for name, model in [
            ("Item-CF",  _item_cf),
            ("User-CF",  _user_cf),
            ("SVD",      _svd),
            ("Content",  _cb),
            ("Hybrid",   _hybrid),
        ]:
            y_true = _test["rating"].to_numpy()
            y_pred = model.predict_batch(_test)
            results[name] = {
                "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            }
        return results

    with st.spinner("Đang đánh giá trên test set..."):
        metrics = compute_metrics(item_cf, user_cf, svd, cb, hybrid, test)

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    colors_model = {
        "Item-CF":  "#4e8ef7",
        "User-CF":  "#f76e6e",
        "SVD":      "#43c78a",
        "Content":  "#f7a84e",
        "Hybrid":   "#b06ef7",
    }
    best_mae = min(metrics, key=lambda k: metrics[k]["MAE"])

    for col, (name, m) in zip([col1, col2, col3, col4, col5], metrics.items()):
        crown = " 👑" if name == best_mae else ""
        with col:
            st.markdown(f"""
            <div class="metric-box" style="border-color:{colors_model[name]}">
                <div class="label">{name}{crown}</div>
                <div class="value">MAE {m['MAE']}</div>
                <div style="color:#aaa;font-size:13px">RMSE {m['RMSE']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Biểu đồ so sánh
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_names = list(metrics.keys())
    bar_colors  = [colors_model[n] for n in model_names]

    for ax, metric_key in zip(axes, ["MAE", "RMSE"]):
        vals = [metrics[n][metric_key] for n in model_names]
        bars = ax.bar(model_names, vals, color=bar_colors, edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_title(f"So sánh {metric_key}", fontsize=13, fontweight="bold")
        ax.set_ylabel(f"{metric_key} (thấp hơn = tốt hơn)")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Hiệu suất trên Test Set (80/20 split)", fontsize=12, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    # Bảng so sánh đặc điểm
    st.subheader("📝 Phân tích")
    st.markdown(f"""
| Tiêu chí | Item-CF | User-CF | SVD | Content-Based | **Hybrid** |
|----------|:---:|:---:|:---:|:---:|:---:|
| Cơ sở | Rating pattern phim | Rating pattern user | Latent factors | Genres + Tags | CF + CB |
| Cold Start Item | ❌ | ❌ | ❌ | ✅ | ✅ |
| Cold Start User | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| Scalability | Tốt | Kém | Tốt nhất | Tốt | Tốt |
| Explainability | Cao | Cao | Thấp | Cao | Trung bình |
| MAE | {metrics['Item-CF']['MAE']} | {metrics['User-CF']['MAE']} | {metrics['SVD']['MAE']} | {metrics['Content']['MAE']} | **{metrics['Hybrid']['MAE']}** |
| RMSE | {metrics['Item-CF']['RMSE']} | {metrics['User-CF']['RMSE']} | {metrics['SVD']['RMSE']} | {metrics['Content']['RMSE']} | **{metrics['Hybrid']['RMSE']}** |

→ **{best_mae}** cho kết quả MAE thấp nhất.  
→ **Hybrid** kết hợp ưu điểm của cả CF lẫn Content-Based, giải quyết Cold Start triệt để hơn.
    """)

    # Ranking Metrics 
    st.divider()
    st.subheader("Ranking Metrics — HitRate · Precision · Recall · NDCG")
    st.caption(
        "MAE/RMSE đo *sai số dự đoán rating*. "
        "Ranking metrics đo *chất lượng danh sách gợi ý* — thực tế hơn khi deploy."
    )

    K_demo = st.select_slider("Chọn K", options=[5, 10, 20], value=10)

    @st.cache_data(show_spinner="Đang tính ranking metrics (có thể mất 1-2 phút)...")
    def compute_ranking(_item_cf, _user_cf, _svd, _cb, _hybrid, _test, _train_matrix, k):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from evaluate import compute_ranking_metrics

        result = {}
        for name, model in [
            ("Item-CF", _item_cf),
            ("User-CF", _user_cf),
            ("SVD",     _svd),
            ("Content", _cb),
            ("Hybrid",  _hybrid),
        ]:
            m = compute_ranking_metrics(model, _test, _train_matrix, k=k, max_users=100)
            result[name] = m
        return result

    with st.spinner(f"Đang tính Ranking Metrics @ K={K_demo} (sample 100 users)..."):
        rank_metrics = compute_ranking(item_cf, user_cf, svd, cb, hybrid, test, train_matrix, K_demo)

    if rank_metrics:
        metric_names = [f"HitRate@{K_demo}", f"Precision@{K_demo}",
                        f"Recall@{K_demo}", f"NDCG@{K_demo}"]
        model_names3 = list(rank_metrics.keys())
        colors3      = ["#4e8ef7", "#f76e6e", "#43c78a", "#f7a84e", "#b06ef7"]

        # Giải thích từng metric
        with st.expander("Ý nghĩa từng metric"):
            st.markdown(f"""
| Metric | Ý nghĩa |
|--------|---------|
| **HitRate@{K_demo}** | Tỉ lệ user có **ít nhất 1** phim tốt (≥4★) trong top-{K_demo} gợi ý |
| **Precision@{K_demo}** | Trong top-{K_demo} gợi ý, trung bình bao nhiêu % là phim tốt |
| **Recall@{K_demo}** | Trong tất cả phim tốt của user, bao nhiêu % được tìm thấy |
| **NDCG@{K_demo}** | Phim tốt xếp càng *đầu* thì điểm càng cao (penalty vị trí thấp) |
            """)

        fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
        for ax, mkey in zip(axes2, metric_names):
            vals  = [rank_metrics[m].get(mkey, 0) for m in model_names3]
            bars2 = ax.bar(model_names3, vals, color=colors3, edgecolor="white", width=0.5)
            for bar, val in zip(bars2, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax.set_title(mkey, fontweight="bold", fontsize=11)
            ax.set_ylim(0, max(vals) * 1.35 if max(vals) > 0 else 0.1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticklabels(model_names3, fontsize=9)

        plt.suptitle(f"Ranking Metrics @ K={K_demo}  (threshold ≥ 4★, sample 100 users)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)

        # Tìm winner theo NDCG
        best_ndcg = max(rank_metrics, key=lambda m: rank_metrics[m].get(f"NDCG@{K_demo}", 0))
        best_hr   = max(rank_metrics, key=lambda m: rank_metrics[m].get(f"HitRate@{K_demo}", 0))
        st.success(
            f"**NDCG@{K_demo}** tốt nhất: **{best_ndcg}**  |  "
            f"**HitRate@{K_demo}** tốt nhất: **{best_hr}**"
        )


# TAB 3 — Tìm phim tương tự

elif tab_choice == "Tìm phim tương tự":
    st.title("Tìm phim tương tự (Item Similarity)")

    movie_list = all_movies["title"].tolist()
    selected_title = st.selectbox("Chọn một bộ phim", movie_list)
    top_k = st.slider("Số phim tương tự muốn xem", 5, 20, 10)

    selected_id = all_movies[all_movies["title"] == selected_title]["movieId"].values[0]

    # Thông tin phim đã chọn
    st.subheader(f"📽️ {selected_title}")
    genres = all_movies[all_movies["movieId"] == selected_id]["genres"].values[0]
    n_ratings = train_matrix[selected_id].notna().sum() if selected_id in train_matrix.columns else 0
    avg_rating = train_matrix[selected_id].mean() if selected_id in train_matrix.columns else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Thể loại", genres.replace("|", " · "))
    c2.metric("Số lượt rating (train)", n_ratings)
    c3.metric("Rating TB (train)", f"{avg_rating:.2f}" if avg_rating else "N/A")

    st.divider()

    # Tính similarity từ Item-CF model
    if selected_id not in item_cf.item_sim.index:
        st.warning("Phim này không có trong tập train, không tính được similarity.")
    else:
        sim_series = item_cf.item_sim.loc[selected_id].drop(index=selected_id, errors="ignore")
        top_similar = sim_series.nlargest(top_k).reset_index()
        top_similar.columns = ["movieId", "similarity"]
        top_similar = top_similar.merge(movies[["movieId", "title", "genres"]], on="movieId")

        # Biểu đồ
        fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.45)))
        cmap   = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_similar)))
        short  = top_similar["title"].str[:42]
        bars   = ax.barh(range(len(top_similar)), top_similar["similarity"], color=cmap)
        ax.set_yticks(range(len(top_similar)))
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Cosine Similarity")
        ax.set_title(f"Top-{top_k} Phim Tương Tự với\n'{selected_title[:45]}'",
                     fontsize=12, fontweight="bold")
        for bar, val in zip(bars, top_similar["similarity"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        plt.tight_layout()
        st.pyplot(fig)

        # Bảng
        st.subheader("Danh sách phim tương tự")
        top_similar["rank"] = range(1, len(top_similar) + 1)
        top_similar["similarity"] = top_similar["similarity"].round(4)
        st.dataframe(
            top_similar[["rank", "title", "genres", "similarity"]]
            .rename(columns={"rank": "#", "title": "Phim", "genres": "Thể loại", "similarity": "Similarity"}),
            use_container_width=True,
            hide_index=True,
        )

        st.info("💡 Similarity được tính bằng **Cosine Similarity** trên ma trận rating (rating pattern của người dùng với hai phim).")



# TAB 4 — User Mới (Cold Start Demo)
elif tab_choice == "User Mới (Cold Start)":

    # Hàm Cold-Start Inference 
    def cold_start_recommend(model: ItemItemCF, user_ratings_dict: dict, n: int = 10):
        """
        Item-Item CF Fold-in — gợi ý cho user mới không có trong train.

        Không retrain toàn bộ ma trận. Chỉ dùng item_sim đã có:
          - Với mỗi phim chưa xem, tính weighted avg similarity với các phim đã chấm.
          - Hoàn toàn vectorized bằng numpy: 1 lần matrix lookup duy nhất.
        """
        rated_ids = [m for m in user_ratings_dict if m in model.item_sim.columns]
        if not rated_ids:
            return []

        rated_vals = np.array([user_ratings_dict[m] for m in rated_ids], dtype=float)
        unseen     = model.item_sim.index.difference(rated_ids)
        if unseen.empty:
            return []

        # (n_unseen × n_rated) — 1 lần lookup
        sim_mat = model.item_sim.loc[unseen, rated_ids].to_numpy().copy()

        # Giữ top-N neighbors, zero-out phần còn lại
        n_nb = min(model.n_neighbors, len(rated_ids))
        if n_nb < len(rated_ids):
            cutoff  = np.partition(sim_mat, -n_nb, axis=1)[:, -n_nb : -n_nb + 1]
            sim_mat = np.where(sim_mat >= cutoff, sim_mat, 0.0)

        denom = np.abs(sim_mat).sum(axis=1)
        numer = sim_mat @ rated_vals
        mask  = denom > 0
        preds = np.where(mask, numer / np.where(mask, denom, 1.0), model.global_mean)
        preds = np.clip(preds, 0.5, 5.0)

        top_pos = np.argpartition(preds, -min(n, len(preds)))[-n:]
        top_pos = top_pos[np.argsort(preds[top_pos])[::-1]]
        return list(zip(unseen[top_pos].tolist(), preds[top_pos].tolist()))

    # UI 
    st.title(" Thử nghiệm với User Mới")
    st.markdown(
        "Bạn là một **user hoàn toàn mới** — chưa có trong database. "
        "Hãy chấm điểm vài bộ phim bên dưới, hệ thống sẽ gợi ý ngay lập tức "
        "bằng **Item-Item CF Fold-in** (không cần retrain!)."
    )
    st.info("💡 **Cold Start Inference**: thay vì học lại toàn bộ ma trận, "
            "ta chỉ tính weighted similarity giữa phim mới và các phim bạn đã chấm. "
            "Kết quả có ngay trong vài mili-giây.")

    st.divider()

    # Chọn bộ phim hiển thị: top phổ biến + có thể lọc theo thể loại
    all_genres_raw = movies["genres"].dropna().str.split("|").explode().unique()
    all_genres     = sorted([g for g in all_genres_raw if g != "(no genres listed)"])

    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        genre_filter = st.multiselect(
            "Lọc theo thể loại (để trống = tất cả)",
            all_genres,
            default=[],
        )
    with col_filter2:
        n_display = st.slider("Số phim hiển thị", 6, 24, 12, 3)

    # Lấy top phim phổ biến trong train matrix
    popularity = train_matrix.notna().sum().sort_values(ascending=False)
    popular_ids = popularity.index.tolist()

    popular_movies = movies[movies["movieId"].isin(popular_ids)].copy()
    popular_movies["popularity"] = popular_movies["movieId"].map(
        popularity.to_dict()
    ).fillna(0)
    popular_movies = popular_movies.sort_values("popularity", ascending=False)

    if genre_filter:
        mask = popular_movies["genres"].apply(
            lambda g: any(genre in str(g).split("|") for genre in genre_filter)
        )
        popular_movies = popular_movies[mask]

    display_movies = popular_movies.head(n_display).reset_index(drop=True)

    # Lưới chấm điểm
    st.subheader("⭐ Chấm điểm các bộ phim")
    st.caption("Kéo slider để chấm. Để **0 = Chưa xem** — những phim này sẽ không được tính.")

    STAR_OPTIONS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    STAR_LABELS  = {
        0: "Chưa xem", 0.5: "⭐ 0.5", 1.0: "⭐ 1.0", 1.5: "⭐ 1.5",
        2.0: "⭐⭐ 2.0", 2.5: "⭐⭐ 2.5", 3.0: "⭐⭐⭐ 3.0", 3.5: "⭐⭐⭐ 3.5",
        4.0: "⭐⭐⭐⭐ 4.0", 4.5: "⭐⭐⭐⭐ 4.5", 5.0: "⭐⭐⭐⭐⭐ 5.0",
    }

    user_ratings_input = {}   # {movieId: rating}
    cols_per_row = 3
    rows = [display_movies.iloc[i : i + cols_per_row]
            for i in range(0, len(display_movies), cols_per_row)]

    for row_movies in rows:
        cols = st.columns(cols_per_row)
        for col, (_, mrow) in zip(cols, row_movies.iterrows()):
            mid   = int(mrow["movieId"])
            title = mrow["title"]
            genre = mrow["genres"].replace("|", " · ") if pd.notna(mrow["genres"]) else ""
            pop   = int(mrow["popularity"])

            with col:
                st.markdown(
                    f"**{title[:38]}{'…' if len(title) > 38 else ''}**  \n"
                    f"<span style='color:#888;font-size:12px'>{genre[:50]}</span>  \n"
                    f"<span style='color:#555;font-size:11px'>👥 {pop} lượt</span>",
                    unsafe_allow_html=True,
                )
                rating_val = st.select_slider(
                    label="rating",
                    options=STAR_OPTIONS,
                    value=0,
                    format_func=lambda x: STAR_LABELS[x],
                    key=f"rate_{mid}",
                    label_visibility="collapsed",
                )
                if rating_val > 0:
                    user_ratings_input[mid] = rating_val

    # Tóm tắt & nút gợi ý 
    st.divider()
    n_rated = len(user_ratings_input)
    status_col, btn_col = st.columns([3, 1])

    with status_col:
        if n_rated == 0:
            st.warning("Hãy chấm ít nhất 1 bộ phim để nhận gợi ý.")
        elif n_rated < 3:
            st.warning(f"Đã chấm **{n_rated}** phim. Chấm thêm để kết quả chính xác hơn!")
        else:
            st.success(f"Đã chấm **{n_rated}** phim — sẵn sàng gợi ý!")

    with btn_col:
        run_btn = st.button(
            "🎬 Gợi ý ngay!",
            disabled=(n_rated == 0),
            use_container_width=True,
            type="primary",
        )

    # Kết quả
    if run_btn and n_rated > 0:
        with st.spinner("Đang tính toán..."):
            recs = cold_start_recommend(item_cf, user_ratings_input, n=n_recs)

        if not recs:
            st.error("Không tìm được gợi ý phù hợp. Thử chấm thêm phim khác")
        else:
            rec_df = (
                pd.DataFrame(recs, columns=["movieId", "pred_rating"])
                .merge(movies[["movieId", "title", "genres"]], on="movieId")
            )
            rec_df["rank"] = range(1, len(rec_df) + 1)

            st.divider()
            st.subheader(f"Top-{len(rec_df)} phim dành riêng cho bạn")

            # Phim bạn đã chấm (để reference)
            with st.expander("Phim bạn vừa chấm điểm"):
                rated_info = (
                    pd.DataFrame(
                        [(mid, rat) for mid, rat in user_ratings_input.items()],
                        columns=["movieId", "your_rating"],
                    )
                    .merge(movies[["movieId", "title", "genres"]], on="movieId")
                    .sort_values("your_rating", ascending=False)
                )
                rated_info["your_rating"] = rated_info["your_rating"].apply(
                    lambda x: "⭐" * int(x) + (".5" if x % 1 else "")
                )
                st.dataframe(
                    rated_info[["title", "genres", "your_rating"]].rename(
                        columns={"title": "Phim", "genres": "Thể loại", "your_rating": "Bạn chấm"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # Biểu đồ kết quả
            fig, ax = plt.subplots(figsize=(10, max(4, len(rec_df) * 0.5)))
            cmap  = plt.cm.RdYlGn(np.linspace(0.35, 0.9, len(rec_df)))
            short = rec_df["title"].str[:42]
            bars  = ax.barh(range(len(rec_df)), rec_df["pred_rating"], color=cmap)
            ax.set_yticks(range(len(rec_df)))
            ax.set_yticklabels(short, fontsize=9)
            ax.set_xlabel("Predicted Rating ⭐")
            ax.set_title(
                f"Top-{len(rec_df)} Gợi Ý cho User Mới (Cold Start — Item-Item CF Fold-in)",
                fontsize=12, fontweight="bold",
            )
            for bar, val in zip(bars, rec_df["pred_rating"]):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 5.8)
            ax.axvline(4.0, color="gray", ls="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Bảng chi tiết
            rec_df["pred_rating"] = rec_df["pred_rating"].round(2)
            st.dataframe(
                rec_df[["rank", "title", "genres", "pred_rating"]].rename(
                    columns={"rank": "#", "title": "Phim", "genres": "Thể loại", "pred_rating": "⭐ Dự đoán"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Giải thích ngắn
            st.markdown("""
---
**📐 Cách tính (Item-Item CF Fold-in):**

Với mỗi phim chưa xem, dự đoán rating bằng:

$$\\hat{r} = \\frac{\\sum_{j \\in N(i)} sim(i,j) \\cdot r_j}{\\sum_{j \\in N(i)} |sim(i,j)|}$$

Trong đó $N(i)$ là top-K phim tương tự nhất với phim $i$ mà bạn đã chấm, $r_j$ là rating bạn tự chấm.  
**Không cần retrain** — chỉ tra cứu ma trận similarity đã tính sẵn lúc khởi động.
            """)


# Footer
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {len(ratings):,} ratings · {ratings['userId'].nunique()} users · {ratings['movieId'].nunique()} movies")