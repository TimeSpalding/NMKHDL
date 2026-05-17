import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import (
    RATINGS_PATH, MOVIES_PATH, TAGS_PATH, LINKS_PATH,
    SAMPLE_SIZE, RANDOM_STATE, MIN_USER_RATINGS, REPORT_DIR
)


def load_raw_data():
    ratings_full = pd.read_csv(RATINGS_PATH)
    movies       = pd.read_csv(MOVIES_PATH)
    tags         = pd.read_csv(TAGS_PATH)
    links        = pd.read_csv(LINKS_PATH)

    print(f"ratings_full shape: {ratings_full.shape}")
    print(f"movies shape:       {movies.shape}")
    print(f"tags shape:         {tags.shape}")
    print(f"links shape:        {links.shape}")

    return ratings_full, movies, tags, links


def sample_and_clean(ratings_full: pd.DataFrame) -> pd.DataFrame:
    # Lấy mẫu
    ratings = ratings_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    print(f"\nDataset sau khi lấy mẫu: {ratings.shape}")
    print(f"  Số users : {ratings['userId'].nunique()}")
    print(f"  Số movies: {ratings['movieId'].nunique()}")

    # Thống kê mô tả
    print("\n=== Thống kê cột rating ===")
    print(ratings["rating"].describe())

    print("\n=== Kiểm tra giá trị thiếu ===")
    print(ratings.isnull().sum())

    # Phát hiện & lọc ngoại lai
    user_rating_count  = ratings.groupby("userId")["rating"].count()
    movie_rating_count = ratings.groupby("movieId")["rating"].count()

    print(f"\nSố user rating < {MIN_USER_RATINGS} lần : {(user_rating_count < MIN_USER_RATINGS).sum()}")
    print(f"Số phim được đánh giá < 3 lần          : {(movie_rating_count < 3).sum()}")

    valid_users  = user_rating_count[user_rating_count >= MIN_USER_RATINGS].index
    ratings_clean = ratings[ratings["userId"].isin(valid_users)].reset_index(drop=True)

    print(f"\n Sau khi lọc ngoại lai: {ratings_clean.shape[0]} ratings")
    print(f"   Số users : {ratings_clean['userId'].nunique()}")
    print(f"   Số movies: {ratings_clean['movieId'].nunique()}")

    return ratings_clean


def load_data():
    os.makedirs(REPORT_DIR, exist_ok=True)
    ratings_full, movies, tags, links = load_raw_data()
    ratings = sample_and_clean(ratings_full)
    return ratings, movies, tags, links
