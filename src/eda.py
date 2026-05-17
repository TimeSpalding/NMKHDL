"""
eda.py — P2 & P3: Khám phá dữ liệu (EDA) + Utility Matrix + Biểu đồ 1-4
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import REPORT_DIR

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")


def _save(fname: str):
    path = os.path.join(REPORT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  💾 Đã lưu: {path}")


# Biểu đồ 1: Histogram phân phối rating
def plot_rating_distribution(ratings):
    fig, ax = plt.subplots(figsize=(10, 5))
    rating_counts = ratings["rating"].value_counts().sort_index()
    bars = ax.bar(rating_counts.index, rating_counts.values,
                  width=0.4, color="steelblue", edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Điểm Rating (sao)", fontsize=13)
    ax.set_ylabel("Số lượng đánh giá", fontsize=13)
    ax.set_title("Biểu đồ 1: Phân phối Rating trong Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(rating_counts.index)
    plt.tight_layout()
    _save("bieu_do_1_histogram_rating.png")
    plt.show()

    print("\n📝 Nhận xét:")
    print(f"  - Rating trung bình : {ratings['rating'].mean():.2f} sao")
    print(f"  - Rating phổ biến nhất: {ratings['rating'].mode()[0]} sao")


# Biểu đồ 2: Boxplot số lần rating theo user & movie
def plot_rating_boxplot(ratings):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    user_counts  = ratings.groupby("userId")["rating"].count()
    movie_counts = ratings.groupby("movieId")["rating"].count()

    for ax, counts, color, label in zip(
        axes,
        [user_counts, movie_counts],
        ["lightblue", "lightsalmon"],
        ["Users", "Movies"],
    ):
        bp = ax.boxplot(counts, patch_artist=True,
                        boxprops=dict(facecolor=color),
                        medianprops=dict(color="red" if label == "Users" else "blue", linewidth=2))
        ax.set_title(f"Số lần rating theo {label}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Số ratings")
        ax.set_xticklabels([label])
        ax.annotate(f"Median: {counts.median():.0f}\nMax: {counts.max()}",
                    xy=(1.15, counts.median()), fontsize=10, color="gray")

    plt.suptitle("Biểu đồ 2: Phân phối số lượng Rating theo User và Movie",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save("bieu_do_2_boxplot.png")
    plt.show()


# Biểu đồ 3: Heatmap Utility Matrix (20×20)
def build_utility_matrix(ratings):
    utility_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    n_users, n_movies = utility_matrix.shape
    n_ratings   = ratings.shape[0]
    total_cells = n_users * n_movies
    sparsity    = (1 - n_ratings / total_cells) * 100

    print("=== Utility Matrix ===")
    print(f"Kích thước : {n_users} users × {n_movies} movies")
    print(f"Tổng ô     : {total_cells:,}")
    print(f"Ô có giá trị: {n_ratings:,}")
    print(f"🔢 Sparsity  = {sparsity:.2f}%")

    return utility_matrix, sparsity


def plot_utility_heatmap(ratings, utility_matrix, sparsity):
    top_users  = ratings.groupby("userId").count()["rating"].nlargest(20).index
    top_movies = ratings.groupby("movieId").count()["rating"].nlargest(20).index
    subset = utility_matrix.loc[
        utility_matrix.index.isin(top_users),
        utility_matrix.columns.isin(top_movies),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(subset, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                annot=True, fmt=".1f", annot_kws={"size": 8},
                cbar_kws={"label": "Rating (sao)"},
                mask=subset.isnull(), ax=ax)
    ax.set_title("Biểu đồ 3: Utility Matrix — Top 20 Users × Top 20 Movies\n(ô trắng = chưa đánh giá)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Movie ID", fontsize=11)
    ax.set_ylabel("User ID", fontsize=11)
    plt.tight_layout()
    _save("bieu_do_3_heatmap_utility.png")
    plt.show()

    print(f"\n📝 Sparsity = {sparsity:.2f}%: hơn {sparsity:.0f}% ô trong ma trận là trống")


# Biểu đồ 4: Top-10 phim được đánh giá nhiều nhất 
def plot_top10_movies(ratings, movies):
    top10 = (
        ratings.groupby("movieId")["rating"]
        .agg(count="count", mean="mean")
        .nlargest(10, "count")
        .reset_index()
        .merge(movies[["movieId", "title"]], on="movieId")
    )
    top10["short_title"] = top10["title"].str[:35]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top10)), top10["count"],
                   color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["short_title"], fontsize=10)
    ax.set_xlabel("Số lượt đánh giá", fontsize=12)
    ax.set_title("Biểu đồ 4: Top 10 Phim Được Đánh Giá Nhiều Nhất",
                 fontsize=14, fontweight="bold")
    for bar, (_, row) in zip(bars, top10.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row['count']} lượt | ⭐{row['mean']:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, top10["count"].max() * 1.3)
    plt.tight_layout()
    _save("bieu_do_4_top10_movies.png")
    plt.show()


def run_eda(ratings, movies):
    print("\n" + "=" * 50)
    print("P2 — KHÁM PHÁ & LÀM SẠCH DỮ LIỆU")
    print("=" * 50)
    plot_rating_distribution(ratings)
    plot_rating_boxplot(ratings)

    print("\n" + "=" * 50)
    print("P3 — UTILITY MATRIX & SPARSITY")
    print("=" * 50)
    utility_matrix, sparsity = build_utility_matrix(ratings)
    plot_utility_heatmap(ratings, utility_matrix, sparsity)
    plot_top10_movies(ratings, movies)

    return utility_matrix, sparsity
