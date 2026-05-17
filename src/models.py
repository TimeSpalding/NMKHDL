import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import N_NEIGHBORS, SVD_FACTORS


# Item-Item Collaborative Filtering
class ItemItemCF:
    """Item-based CF dùng cosine similarity trên cột phim."""

    def __init__(self, train_matrix: pd.DataFrame, n_neighbors: int = N_NEIGHBORS):
        self.matrix      = train_matrix
        self.n_neighbors = n_neighbors
        self.global_mean = train_matrix.stack().mean()

        # Coi các phim chưa đánh giá là 0 để tính Cosine Similarity chuẩn xác
        filled = train_matrix.fillna(0)
        sim = cosine_similarity(filled.T)  # shape: n_movies × n_movies
        self.item_sim = pd.DataFrame(sim,
                                     index=train_matrix.columns,
                                     columns=train_matrix.columns)

    def predict_rating(self, user_id, movie_id) -> float:
        if user_id not in self.matrix.index:
            return self.global_mean
        if movie_id not in self.matrix.columns:
            return self.global_mean

        # Phim user đã đánh giá
        user_ratings = self.matrix.loc[user_id].dropna()
        rated_movies = user_ratings.index.tolist()

        if movie_id not in self.item_sim.index or not rated_movies:
            return self.global_mean

        # Lấy N neighbors tương đồng nhất (loại chính nó)
        sims = self.item_sim.loc[movie_id, rated_movies].drop(
            labels=[movie_id], errors="ignore"
        )
        top_sims = sims.nlargest(self.n_neighbors)
        if top_sims.sum() == 0:
            return self.global_mean

        weighted = sum(top_sims[m] * user_ratings[m] for m in top_sims.index)
        pred = weighted / top_sims.abs().sum()
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        result    = np.full(len(test_df), self.global_mean)
        test_reset = test_df.reset_index(drop=True)

        for user_id, group in test_reset.groupby("userId"):
            if user_id not in self.matrix.index:
                continue
            user_ratings = self.matrix.loc[user_id].dropna()
            if user_ratings.empty:
                continue

            # Chỉ giữ rated movies có trong similarity matrix
            rated = user_ratings.index.intersection(self.item_sim.columns)
            if rated.empty:
                continue

            valid_pairs = [(m, i) for m, i in zip(group["movieId"], group.index)
                           if m in self.item_sim.index]
            if not valid_pairs:
                continue
            valid_movies, valid_pos = zip(*valid_pairs)

            # 1 lần lookup duy nhất: (n_target × n_rated)
            rated_ratings = user_ratings[rated].to_numpy()
            sim_mat       = self.item_sim.loc[list(valid_movies), rated].to_numpy().copy()

            # Xóa self-similarity
            rated_list = rated.tolist()
            for k, mid in enumerate(valid_movies):
                if mid in rated_list:
                    sim_mat[k, rated_list.index(mid)] = 0.0

            # Zero-out tất cả trừ top-N neighbors
            n_nb = min(self.n_neighbors, sim_mat.shape[1])
            if n_nb < sim_mat.shape[1]:
                cutoff = np.partition(sim_mat, -n_nb, axis=1)[:, -n_nb : -n_nb + 1]
                sim_mat = np.where(sim_mat >= cutoff, sim_mat, 0.0)

            # Weighted average hoàn toàn bằng numpy
            denom = np.abs(sim_mat).sum(axis=1)
            numer = sim_mat @ rated_ratings
            mask  = denom > 0
            preds = np.where(mask, numer / np.where(mask, denom, 1.0), self.global_mean)

            for pos, pred in zip(valid_pos, np.clip(preds, 0.5, 5.0)):
                result[pos] = pred

        return result

    def recommend(self, user_id, n: int = 10):
        if user_id not in self.matrix.index:
            return []
        seen   = self.matrix.loc[user_id].dropna().index
        unseen = [m for m in self.matrix.columns if m not in seen]
        preds  = [(m, self.predict_rating(user_id, m)) for m in unseen]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]


# User-User Collaborative Filtering 
    def __init__(self, train_matrix: pd.DataFrame, n_neighbors: int = N_NEIGHBORS):
        self.matrix      = train_matrix
        self.n_neighbors = n_neighbors
        self.global_mean = train_matrix.stack().mean()

        filled = train_matrix.fillna(0)
        sim = cosine_similarity(filled)  # shape: n_users × n_users
        self.user_sim = pd.DataFrame(sim,
                                     index=train_matrix.index,
                                     columns=train_matrix.index)

    def predict_rating(self, user_id, movie_id) -> float:
        if user_id not in self.matrix.index:
            return self.global_mean
        if movie_id not in self.matrix.columns:
            return self.global_mean

        # Lấy users đã đánh giá phim này
        movie_ratings = self.matrix[movie_id].dropna()
        raters = movie_ratings.index.tolist()

        if user_id not in self.user_sim.index or not raters:
            return self.global_mean

        sims = self.user_sim.loc[user_id, raters].drop(
            labels=[user_id], errors="ignore"
        )
        top_sims = sims.nlargest(self.n_neighbors)
        if top_sims.sum() == 0:
            return self.global_mean

        weighted = sum(top_sims[u] * movie_ratings[u] for u in top_sims.index)
        pred = weighted / top_sims.abs().sum()
        return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        result     = np.full(len(test_df), self.global_mean)
        test_reset = test_df.reset_index(drop=True)

        for movie_id, group in test_reset.groupby("movieId"):
            if movie_id not in self.matrix.columns:
                continue
            movie_ratings = self.matrix[movie_id].dropna()
            raters = movie_ratings.index.intersection(self.user_sim.columns)
            if raters.empty:
                continue

            valid_pairs = [(u, i) for u, i in zip(group["userId"], group.index)
                           if u in self.user_sim.index]
            if not valid_pairs:
                continue
            valid_users, valid_pos = zip(*valid_pairs)

            # 1 lần lookup: (n_target_users × n_raters)
            rater_ratings = movie_ratings[raters].to_numpy()
            sim_mat       = self.user_sim.loc[list(valid_users), raters].to_numpy().copy()

            # Xóa self-similarity
            raters_list = raters.tolist()
            for k, uid in enumerate(valid_users):
                if uid in raters_list:
                    sim_mat[k, raters_list.index(uid)] = 0.0

            # Zero-out tất cả trừ top-N
            n_nb = min(self.n_neighbors, sim_mat.shape[1])
            if n_nb < sim_mat.shape[1]:
                cutoff = np.partition(sim_mat, -n_nb, axis=1)[:, -n_nb : -n_nb + 1]
                sim_mat = np.where(sim_mat >= cutoff, sim_mat, 0.0)

            denom = np.abs(sim_mat).sum(axis=1)
            numer = sim_mat @ rater_ratings
            mask  = denom > 0
            preds = np.where(mask, numer / np.where(mask, denom, 1.0), self.global_mean)

            for pos, pred in zip(valid_pos, np.clip(preds, 0.5, 5.0)):
                result[pos] = pred

        return result

    def recommend(self, user_id, n: int = 10):
        if user_id not in self.matrix.index:
            return []
        seen   = self.matrix.loc[user_id].dropna().index
        unseen = [m for m in self.matrix.columns if m not in seen]
        preds  = [(m, self.predict_rating(user_id, m)) for m in unseen]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]


# SVD Recommender 
class SVDRecommender:
    def __init__(self, train_matrix: pd.DataFrame, n_factors: int = SVD_FACTORS):
        self.matrix      = train_matrix.copy()
        self.n_factors   = n_factors
        self.global_mean = train_matrix.stack().mean()

        # Điền NaN bằng mean của từng bộ phim để SVD học xu hướng phim
        movie_means = train_matrix.mean()
        matrix_filled = train_matrix.fillna(movie_means).fillna(self.global_mean)

        U, sigma, Vt = np.linalg.svd(matrix_filled.values, full_matrices=False)
        k            = min(n_factors, len(sigma))
        self.U       = U[:, :k]
        self.sigma   = np.diag(sigma[:k])
        self.Vt      = Vt[:k, :]

        self.R_approx = pd.DataFrame(
            self.U @ self.sigma @ self.Vt,
            index=train_matrix.index,
            columns=train_matrix.columns,
        )

        print(f"SVD khởi tạo (k={k}): U{self.U.shape}, Σ{self.sigma.shape}, Vt{self.Vt.shape}")

    def predict_rating(self, user_id, movie_id) -> float:
        if user_id not in self.R_approx.index:
            return self.global_mean
        if movie_id not in self.R_approx.columns:
            return self.global_mean
        return float(np.clip(self.R_approx.loc[user_id, movie_id], 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        result     = np.full(len(test_df), self.global_mean)
        test_reset = test_df.reset_index(drop=True)

        valid = (test_reset["userId"].isin(self.R_approx.index) &
                 test_reset["movieId"].isin(self.R_approx.columns))
        if valid.any():
            sub     = test_reset[valid]
            row_pos = self.R_approx.index.get_indexer(sub["userId"])
            col_pos = self.R_approx.columns.get_indexer(sub["movieId"])
            result[valid.to_numpy()] = np.clip(
                self.R_approx.to_numpy()[row_pos, col_pos], 0.5, 5.0
            )
        return result

    def recommend(self, user_id, n: int = 10):
        if user_id not in self.R_approx.index:
            return []
        seen   = self.matrix.loc[user_id].dropna().index
        unseen = self.R_approx.columns.difference(seen)
        if unseen.empty:
            return []
        preds   = np.clip(self.R_approx.loc[user_id, unseen].to_numpy(), 0.5, 5.0)
        top_pos = np.argpartition(preds, -min(n, len(preds)))[-n:]
        top_pos = top_pos[np.argsort(preds[top_pos])[::-1]]
        return list(zip(unseen[top_pos].tolist(), preds[top_pos].tolist()))


# Content-Based Filtering 
class ContentBasedCF(ItemItemCF):
    def __init__(self, movies: pd.DataFrame, train_matrix: pd.DataFrame, tags_df: pd.DataFrame = None, n_neighbors: int = N_NEIGHBORS):
        self.matrix      = train_matrix
        self.n_neighbors = n_neighbors
        self.global_mean = train_matrix.stack().mean()

        movie_content = movies[['movieId', 'genres']].copy()
        movie_content['genres'] = movie_content['genres'].fillna('').str.replace('|', ' ')

        if tags_df is not None and not tags_df.empty:
            tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
            movie_content = movie_content.merge(tags_grouped, on='movieId', how='left')
            movie_content['tag'] = movie_content['tag'].fillna('')
            movie_content['content'] = movie_content['genres'] + ' ' + movie_content['tag']
        else:
            movie_content['content'] = movie_content['genres']

        valid_movies = train_matrix.columns.intersection(movie_content['movieId'])
        mc = movie_content.set_index('movieId').loc[valid_movies]

        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(mc['content'])
        sim = cosine_similarity(tfidf_matrix)

        self.item_sim = pd.DataFrame(sim, index=valid_movies, columns=valid_movies)


# Hybrid Recommender
class HybridRecommender:
    def __init__(self, model1, model2, alpha=0.7):
        self.model1 = model1
        self.model2 = model2
        self.alpha  = alpha
        self.matrix = model1.matrix

    def predict_rating(self, user_id, movie_id) -> float:
        p1 = self.model1.predict_rating(user_id, movie_id)
        p2 = self.model2.predict_rating(user_id, movie_id)
        return float(np.clip(self.alpha * p1 + (1 - self.alpha) * p2, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        p1 = self.model1.predict_batch(test_df)
        p2 = self.model2.predict_batch(test_df)
        return np.clip(self.alpha * p1 + (1 - self.alpha) * p2, 0.5, 5.0)

    def recommend(self, user_id, n: int = 10):
        if user_id not in self.matrix.index:
            return []
        seen   = self.matrix.loc[user_id].dropna().index
        unseen = self.matrix.columns.difference(seen)
        if unseen.empty:
            return []

        # Vectorized inference
        test_df = pd.DataFrame({'userId': user_id, 'movieId': unseen})
        preds = self.predict_batch(test_df)

        top_pos = np.argpartition(preds, -min(n, len(preds)))[-n:]
        top_pos = top_pos[np.argsort(preds[top_pos])[::-1]]
        return list(zip(unseen[top_pos].tolist(), preds[top_pos].tolist()))
