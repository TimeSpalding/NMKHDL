import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from config import REPORT_DIR, TEST_SIZE, RANDOM_STATE, KFOLD_SPLITS, N_NEIGHBORS, SVD_FACTORS, RELEVANCE_THRESHOLD, TOP_K_LIST
from models import ItemItemCF, UserUserCF, SVDRecommender

plt.rcParams["figure.figsize"] = (10, 6)


def _save(fname: str):
    path = os.path.join(REPORT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Đã lưu: {path}")


# Chia train / test 
def split_data(ratings: pd.DataFrame):
    train, test = train_test_split(ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_matrix = train.pivot_table(index="userId", columns="movieId", values="rating")
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Train matrix: {train_matrix.shape[0]} users × {train_matrix.shape[1]} movies")
    return train, test, train_matrix


# Biểu đồ 5: Top-N recommendations cho 1 user 
def plot_top_n_recommendations(model, user_id, movies: pd.DataFrame, title: str, chart_file: str):
    recs = model.recommend(user_id, n=10)
    if not recs:
        print(f"Không có gợi ý cho user {user_id}")
        return

    rec_df = pd.DataFrame(recs, columns=["movieId", "pred_rating"])
    rec_df = rec_df.merge(movies[["movieId", "title"]], on="movieId")
    rec_df["short_title"] = rec_df["title"].str[:40]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(rec_df)))
    bars = ax.barh(range(len(rec_df)), rec_df["pred_rating"], color=colors)
    ax.set_yticks(range(len(rec_df)))
    ax.set_yticklabels(rec_df["short_title"], fontsize=10)
    ax.set_xlabel("Predicted Rating", fontsize=12)
    ax.set_title(f"Biểu đồ 5: Top-10 Gợi Ý ({title}) cho User {user_id}",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, rec_df["pred_rating"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}⭐", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 6)
    plt.tight_layout()
    _save(chart_file)
    plt.show()


# Tính MAE / RMSE 
def _collect_predictions(model, test_data: pd.DataFrame):
    y_true, y_pred = [], []
    for _, row in test_data.iterrows():
        y_true.append(row["rating"])
        y_pred.append(model.predict_rating(row["userId"], row["movieId"]))
    return y_true, y_pred


def evaluate_models(item_cf, user_cf, test_data: pd.DataFrame):
    """Trả về dict metrics cho cả hai mô hình."""
    y_true_i, y_pred_i = _collect_predictions(item_cf, test_data)
    y_true_u, y_pred_u = _collect_predictions(user_cf, test_data)

    metrics = {
        "Item-CF": {
            "mae":  mean_absolute_error(y_true_i, y_pred_i),
            "rmse": np.sqrt(mean_squared_error(y_true_i, y_pred_i)),
        },
        "User-CF": {
            "mae":  mean_absolute_error(y_true_u, y_pred_u),
            "rmse": np.sqrt(mean_squared_error(y_true_u, y_pred_u)),
        },
    }

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 50)
    print(f"{'Mô hình':<20} {'MAE':>8} {'RMSE':>8}")
    print("-" * 40)
    for name, m in metrics.items():
        print(f"{name:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f}")

    better = min(metrics, key=lambda k: metrics[k]["mae"])
    print(f"\n→ {better} cho kết quả tốt hơn theo MAE")
    return metrics


# Biểu đồ 6: So sánh MAE & RMSE 
def plot_model_comparison(metrics: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = list(metrics.keys())
    colors = ["steelblue", "tomato"]

    for ax, metric_key, ylabel in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"]):
        vals = [metrics[m][metric_key] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"So sánh {ylabel}", fontsize=13, fontweight="bold")
        ax.set_ylabel(f"{ylabel} (thấp hơn = tốt hơn)")
        ax.set_ylim(0, max(vals) * 1.3)

    plt.suptitle("Biểu đồ 6: So Sánh Hiệu Suất Hai Mô Hình CF",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("bieu_do_6_model_comparison.png")
    plt.show()


# Biểu đồ 7: Cold Start Analysis 
def plot_cold_start(ratings: pd.DataFrame):
    user_cnt  = ratings.groupby("userId")["rating"].count()
    movie_cnt = ratings.groupby("movieId")["rating"].count()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, counts, bins, labels, colors, title, xlabel in [
        (axes[0], user_cnt,
         [0, 5, 10, 20, 50, 100, user_cnt.max() + 1],
         ["<5 (Cold)", "5-10", "10-20", "20-50", "50-100", ">100"],
         ["#d32f2f"] + ["#90caf9"] * 5,
         "Phân bố User theo số lượng Rating", "Số lần đánh giá"),
        (axes[1], movie_cnt,
         [0, 3, 5, 10, 20, 50, movie_cnt.max() + 1],
         ["<3 (Cold)", "3-5", "5-10", "10-20", "20-50", ">50"],
         ["#d32f2f"] + ["#a5d6a7"] * 5,
         "Phân bố Movie theo số lượng Rating", "Số lần được đánh giá"),
    ]:
        hist, _ = np.histogram(counts, bins=bins)
        bars = ax.bar(labels, hist, color=colors, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Số lượng")
        for bar, val in zip(bars, hist):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.suptitle("Biểu đồ 7: Phân Tích Cold Start Problem",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("bieu_do_7_cold_start.png")
    plt.show()

    cold_u = (user_cnt < 5).sum()
    cold_m = (movie_cnt < 3).sum()
    print(f"Cold Start users  (< 5 ratings) : {cold_u}/{len(user_cnt)} = {cold_u/len(user_cnt)*100:.1f}%")
    print(f"Cold Start movies (< 3 ratings) : {cold_m}/{len(movie_cnt)} = {cold_m/len(movie_cnt)*100:.1f}%")


# K-Fold Cross Validation 
def kfold_evaluation(ratings: pd.DataFrame):
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    print(f"\nĐang thực hiện {KFOLD_SPLITS}-Fold Cross Validation...")

    for fold, (train_idx, test_idx) in enumerate(kf.split(ratings), 1):
        fold_train = ratings.iloc[train_idx]
        fold_test  = ratings.iloc[test_idx]
        fold_matrix = fold_train.pivot_table(index="userId", columns="movieId", values="rating")

        fi_cf = ItemItemCF(fold_matrix, n_neighbors=N_NEIGHBORS)
        fu_cf = UserUserCF(fold_matrix, n_neighbors=N_NEIGHBORS)

        yt_i, yp_i = _collect_predictions(fi_cf, fold_test)
        yt_u, yp_u = _collect_predictions(fu_cf, fold_test)

        mae_i  = mean_absolute_error(yt_i, yp_i)
        rmse_i = np.sqrt(mean_squared_error(yt_i, yp_i))
        mae_u  = mean_absolute_error(yt_u, yp_u)
        rmse_u = np.sqrt(mean_squared_error(yt_u, yp_u))

        fold_results.append({"fold": fold,
                              "item_mae": mae_i, "item_rmse": rmse_i,
                              "user_mae": mae_u, "user_rmse": rmse_u})
        print(f"  Fold {fold}: Item-CF MAE={mae_i:.4f} | User-CF MAE={mae_u:.4f}")

    kdf = pd.DataFrame(fold_results).set_index("fold")
    print("\n" + "=" * 65)
    print("KẾT QUẢ TRUNG BÌNH K-FOLD")
    print("=" * 65)
    for model, m_mae, m_rmse in [
        ("Item-Item CF", "item_mae", "item_rmse"),
        ("User-User CF", "user_mae", "user_rmse"),
    ]:
        print(f"{model:<20}  MAE = {kdf[m_mae].mean():.4f} ± {kdf[m_mae].std():.4f}"
              f"   RMSE = {kdf[m_rmse].mean():.4f} ± {kdf[m_rmse].std():.4f}")

    return kdf


# Biểu đồ 8: K-Fold chart 
def plot_kfold(kdf: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    folds = kdf.index.tolist()

    for ax, (i_col, u_col), ylabel in zip(
        axes,
        [("item_mae", "user_mae"), ("item_rmse", "user_rmse")],
        ["MAE", "RMSE"],
    ):
        ax.plot(folds, kdf[i_col], "o-", color="steelblue", lw=2, ms=7, label="Item-Item CF")
        ax.plot(folds, kdf[u_col], "s--", color="tomato", lw=2, ms=7, label="User-User CF")
        ax.axhline(kdf[i_col].mean(), color="steelblue", ls=":", alpha=0.5)
        ax.axhline(kdf[u_col].mean(), color="tomato", ls=":", alpha=0.5)
        ax.set_title(f"{ylabel} theo từng Fold", fontweight="bold")
        ax.set_xlabel("Fold")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_xticks(folds)

    plt.suptitle("Biểu đồ 8: Kết Quả 5-Fold Cross Validation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("bieu_do_8_kfold.png")
    plt.show()


# Biểu đồ 9: So sánh 3 mô hình (CF + SVD) 
def evaluate_svd_and_compare(svd_model, test_data: pd.DataFrame, metrics_cf: dict):
    y_true, y_pred = _collect_predictions(svd_model, test_data)
    mae_svd  = mean_absolute_error(y_true, y_pred)
    rmse_svd = np.sqrt(mean_squared_error(y_true, y_pred))

    all_metrics = {
        "Item-Item CF": metrics_cf["Item-CF"],
        "User-User CF": metrics_cf["User-CF"],
        f"SVD (k={SVD_FACTORS})": {"mae": mae_svd, "rmse": rmse_svd},
    }

    print("\n" + "=" * 55)
    print("SO SÁNH 3 MÔ HÌNH")
    print("=" * 55)
    print(f"{'Mô hình':<22} {'MAE':>10} {'RMSE':>10}")
    print("-" * 45)
    for name, m in all_metrics.items():
        print(f"{name:<22} {m['mae']:>10.4f} {m['rmse']:>10.4f}")

    best = min(all_metrics, key=lambda k: all_metrics[k]["mae"])
    print(f"\n Mô hình tốt nhất theo MAE: {best}")

    # Biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models3 = list(all_metrics.keys())
    colors3 = ["steelblue", "tomato", "mediumseagreen"]

    for ax, metric_key, ylabel in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"]):
        vals = [all_metrics[m][metric_key] for m in models3]
        bars = ax.bar(models3, vals, color=colors3, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
        ax.set_title(f"So sánh {ylabel} — 3 mô hình", fontweight="bold")
        ax.set_ylabel(f"{ylabel} (thấp hơn = tốt hơn)")
        ax.set_ylim(0, max(vals) * 1.3)

    plt.suptitle("Biểu đồ 9: So Sánh Item-CF vs User-CF vs SVD",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("bieu_do_9_compare_3models.png")
    plt.show()

    return all_metrics



# RANKING METRICS — HitRate@K · Precision@K · Recall@K · NDCG@K
def _ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K cho 1 user.
    Công thức: DCG@K / IDCG@K
      DCG@K  = Σ rel_i / log2(i+2)   (i=0..K-1, rel_i=1 nếu item i là relevant)
      IDCG@K = DCG của ranking lý tưởng (tất cả relevant xếp đầu)
    """
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    n_ideal = min(len(relevant), k)
    idcg    = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_ranking_metrics(
    model,
    test_data: pd.DataFrame,
    train_matrix: pd.DataFrame,
    k: int = 10,
    threshold: float = RELEVANCE_THRESHOLD,
    max_users: int = 200,       # giới hạn để chạy nhanh hơn
) -> dict:
    """
    Tính HitRate@K, Precision@K, Recall@K, NDCG@K trên tập test.

    Định nghĩa:
      relevant(u)     = {phim user u chấm >= threshold trong TEST set}
      recommended(u)  = top-K phim model gợi ý (chưa thấy trong TRAIN)

    Trả về dict: {"HitRate@K": ..., "Precision@K": ..., "Recall@K": ..., "NDCG@K": ...}
    """
    # Phim "tốt" của mỗi user trong test
    user_relevant = (
        test_data[test_data["rating"] >= threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    # Chỉ xét user có trong train và có ít nhất 1 relevant item trong test
    eligible = [u for u in user_relevant if u in train_matrix.index]
    if not eligible:
        print("Không có user nào đủ điều kiện để tính ranking metrics.")
        return {}

    # Giới hạn số user để tránh chạy quá lâu (recommend() tốn O(n_movies) mỗi user)
    if len(eligible) > max_users:
        rng      = np.random.default_rng(RANDOM_STATE)
        eligible = rng.choice(eligible, size=max_users, replace=False).tolist()

    hits, precisions, recalls, ndcgs = [], [], [], []

    for user_id in eligible:
        relevant = user_relevant[user_id]
        recs     = model.recommend(user_id, n=k)
        rec_list = [mid for mid, _ in recs]
        rec_set  = set(rec_list)

        hit_set = rec_set & relevant
        hits.append(1 if hit_set else 0)
        precisions.append(len(hit_set) / k)
        recalls.append(len(hit_set) / len(relevant) if relevant else 0.0)
        ndcgs.append(_ndcg_at_k(rec_list, relevant, k))

    n = len(hits)
    return {
        f"HitRate@{k}":   round(float(np.mean(hits)),       4),
        f"Precision@{k}": round(float(np.mean(precisions)), 4),
        f"Recall@{k}":    round(float(np.mean(recalls)),    4),
        f"NDCG@{k}":      round(float(np.mean(ndcgs)),      4),
        "_n_users":        n,
        "_k":              k,
        "_threshold":      threshold,
    }


def compare_ranking_metrics(
    models: dict,
    test_data: pd.DataFrame,
    train_matrix: pd.DataFrame,
    k_list: list = None,
    threshold: float = RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    So sánh ranking metrics của nhiều mô hình trên nhiều giá trị K.

    models = {"Item-CF": item_cf, "User-CF": user_cf, "SVD": svd}
    Trả về DataFrame: rows = (model, K), cols = HitRate/Precision/Recall/NDCG
    """
    if k_list is None:
        k_list = TOP_K_LIST

    rows = []
    for model_name, model in models.items():
        for k in k_list:
            print(f"  Đang tính {model_name} @ K={k} ...", end=" ", flush=True)
            m = compute_ranking_metrics(model, test_data, train_matrix, k=k, threshold=threshold)
            if not m:
                continue
            rows.append({
                "Model": model_name,
                "K":     k,
                "HitRate":   m[f"HitRate@{k}"],
                "Precision": m[f"Precision@{k}"],
                "Recall":    m[f"Recall@{k}"],
                "NDCG":      m[f"NDCG@{k}"],
                "n_users":   m["_n_users"],
            })
            print(f"NDCG={m[f'NDCG@{k}']:.4f}  HitRate={m[f'HitRate@{k}']:.4f}")

    df = pd.DataFrame(rows)

    # In bảng tổng hợp
    print("\n" + "=" * 70)
    print(f"RANKING METRICS (threshold={threshold}★)")
    print("=" * 70)
    print(df.to_string(index=False))
    return df


# Biểu đồ 10: Ranking Metrics
def plot_ranking_metrics(ranking_df: pd.DataFrame):
    """
    2×2 subplot: HitRate / Precision / Recall / NDCG
    Mỗi subplot: đường theo K, 1 đường mỗi mô hình.
    """
    metrics_cols = ["HitRate", "Precision", "Recall", "NDCG"]
    titles = [
        "HitRate@K  (cao hơn = tốt hơn)\nTỉ lệ user có ít nhất 1 phim đúng trong top-K",
        "Precision@K  (cao hơn = tốt hơn)\nTỉ lệ phim gợi ý thực sự relevant",
        "Recall@K  (cao hơn = tốt hơn)\nTỉ lệ phim relevant được tìm thấy",
        "NDCG@K  (cao hơn = tốt hơn)\nChất lượng xếp hạng (phim tốt xếp càng đầu càng điểm cao)",
    ]
    colors = {"Item-CF": "#4e8ef7", "User-CF": "#f76e6e", "SVD": "#43c78a"}
    markers = {"Item-CF": "o", "User-CF": "s", "SVD": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, col, title in zip(axes, metrics_cols, titles):
        for model_name, grp in ranking_df.groupby("Model"):
            grp = grp.sort_values("K")
            ax.plot(
                grp["K"], grp[col],
                color=colors.get(model_name, "gray"),
                marker=markers.get(model_name, "o"),
                lw=2, ms=8, label=model_name,
            )
            # Annotate điểm cuối
            last = grp.iloc[-1]
            ax.annotate(
                f"{last[col]:.3f}",
                (last["K"], last[col]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=8, color=colors.get(model_name, "gray"),
            )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_ylabel(col)
        ax.set_xticks(ranking_df["K"].unique())
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

    plt.suptitle(
        "Biểu đồ 10: Ranking Metrics — Item-CF vs User-CF vs SVD",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save("bieu_do_10_ranking_metrics.png")
    plt.show()

    # In nhận xét
    best_ndcg_row = ranking_df.loc[ranking_df["NDCG"].idxmax()]
    print(f"\n📝 Nhận xét:")
    print(f"  NDCG cao nhất: {best_ndcg_row['Model']} @ K={int(best_ndcg_row['K'])} "
          f"→ {best_ndcg_row['NDCG']:.4f}")
    print(f"  → Mô hình này xếp hạng phim tốt nhất: phim relevant xuất hiện sớm hơn trong top-K.")

