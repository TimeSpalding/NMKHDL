import sys
import os
import numpy as np

# Để import được các module trong src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import REPORT_DIR
from data_loader import load_data
from eda import run_eda
from models import ItemItemCF, UserUserCF, SVDRecommender, ContentBasedCF, HybridRecommender
from evaluate import (
    split_data,
    plot_top_n_recommendations,
    evaluate_models,
    plot_model_comparison,
    plot_cold_start,
    kfold_evaluation,
    plot_kfold,
    evaluate_svd_and_compare,
    compare_ranking_metrics,
    plot_ranking_metrics,
)
import os; os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("HỆ KHUYẾN NGHỊ PHIM — MovieLens")
    print("=" * 60)

    # P1 & P2: Đọc & làm sạch dữ liệu
    ratings, ratings_full, movies, tags, links = load_data()

    # P2 & P3: EDA + Utility Matrix 
    utility_matrix, sparsity = run_eda(ratings, movies)

    # P4: Chia train/test, huấn luyện mô hình
    print("\n" + "=" * 50)
    print("P4 — XÂY DỰNG MÔ HÌNH COLLABORATIVE FILTERING")
    print("=" * 50)
    train_data, test_data, train_matrix = split_data(ratings)

    print("\nĐang huấn luyện Item-Item CF...")
    item_cf = ItemItemCF(train_matrix)

    print("Đang huấn luyện User-User CF...")
    user_cf = UserUserCF(train_matrix)

    # Biểu đồ 5: Gợi ý cho user đầu tiên trong tập train
    sample_user = train_matrix.index[0]
    plot_top_n_recommendations(item_cf, sample_user, movies,
                                "Item-Item CF", "bieu_do_5a_itemcf_recs.png")
    plot_top_n_recommendations(user_cf, sample_user, movies,
                                "User-User CF", "bieu_do_5b_usercf_recs.png")

    # P5: Đánh giá mô hình 
    print("\n" + "=" * 50)
    print("P5 — ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 50)
    metrics_cf = evaluate_models(item_cf, user_cf, test_data)
    plot_model_comparison(metrics_cf)

    # Bonus 1: Cold Start
    print("\n" + "=" * 50)
    print("BONUS 1 — PHÂN TÍCH COLD START")
    print("=" * 50)
    plot_cold_start(ratings_full)

    # Bonus 2: K-Fold Cross Validation 
    print("\n" + "=" * 50)
    print("BONUS 2 — K-FOLD CROSS VALIDATION")
    print("=" * 50)
    kdf = kfold_evaluation(ratings)
    plot_kfold(kdf)

    # Bonus 3: SVD
    print("\n" + "=" * 50)
    print("BONUS 3 — MATRIX FACTORIZATION (SVD)")
    print("=" * 50)
    svd_model = SVDRecommender(train_matrix)
    evaluate_svd_and_compare(svd_model, test_data, metrics_cf)

    # Bonus 4: Content-Based + Hybrid
    print("\n" + "=" * 50)
    print("BONUS 4 — CONTENT-BASED + HYBRID RECOMMENDER")
    print("=" * 50)
    print("\nĐang xây dựng Content-Based model (TF-IDF)...")
    cb_model = ContentBasedCF(movies, train_matrix, tags_df=tags)

    print("\nĐang xây dựng Hybrid model (0.7 CF + 0.3 CB)...")
    hybrid_model = HybridRecommender(item_cf, cb_model, alpha=0.7)

    # So sánh MAE/RMSE tất cả 5 mô hình
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    print(f"\n{'Mô hình':<25} {'MAE':>8} {'RMSE':>8}")
    print("-" * 45)
    all_5 = {
        "Item-Item CF": item_cf,
        "User-User CF": user_cf,
        "SVD":          svd_model,
        "Content-Based": cb_model,
        "Hybrid (0.7CF+0.3CB)": hybrid_model,
    }
    for name, mdl in all_5.items():
        y_true = test_data["rating"].to_numpy()
        y_pred = mdl.predict_batch(test_data)
        mae    = mean_absolute_error(y_true, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{name:<25} {mae:>8.4f} {rmse:>8.4f}")

    # Bonus 5: Ranking Metrics 
    print("\n" + "=" * 50)
    print("BONUS 5 — RANKING METRICS (HitRate / Precision / Recall / NDCG)")
    print("=" * 50)
    ranking_df = compare_ranking_metrics(
        models={
            "Item-CF":  item_cf,
            "User-CF":  user_cf,
            "SVD":      svd_model,
            "Content":  cb_model,
            "Hybrid":   hybrid_model,
        },
        test_data=test_data,
        train_matrix=train_matrix,
    )
    plot_ranking_metrics(ranking_df)

    print("\n✅ HOÀN THÀNH! Biểu đồ đã lưu tại:", REPORT_DIR)


if __name__ == "__main__":
    main()