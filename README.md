# Hệ Khuyến Nghị Phim — MovieLens (Local)

## Cấu trúc project

```
movie_recommender/
├── data/                  ← đặt 4 file CSV vào đây
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
├── report/                ← biểu đồ tự động lưu ở đây (tạo tự động)
├── src/
│   ├── config.py          ← đường dẫn & tham số
│   ├── data_loader.py     ← P1 & P2: đọc + làm sạch
│   ├── eda.py             ← P2 & P3: biểu đồ 1-4
│   ├── models.py          ← P4: Item-CF, User-CF, SVD
│   └── evaluate.py        ← P5 & Bonus: đánh giá, biểu đồ 5-9
├── main.py                ← chạy toàn bộ pipeline
└── requirements.txt
```

## Cài đặt & chạy

```bash
# 1. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Tải dataset MovieLens
#    https://www.kaggle.com/datasets/tanvirrahmanornob/mymoviedatasetsforrecom
#    Giải nén → đặt 4 file CSV vào thư mục data/

# 4. Chạy
python main.py
```

## Tuỳ chỉnh tham số

Mở `src/config.py` để thay đổi:
| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `SAMPLE_SIZE` | 5000 | Số ratings lấy mẫu |
| `N_NEIGHBORS` | 10 | Số neighbors trong CF |
| `SVD_FACTORS` | 20 | Số latent factors |
| `KFOLD_SPLITS` | 5 | Số fold trong cross-validation |

## Output

Sau khi chạy xong, thư mục `report/` sẽ có 9 biểu đồ:
- `bieu_do_1_histogram_rating.png`
- `bieu_do_2_boxplot.png`
- `bieu_do_3_heatmap_utility.png`
- `bieu_do_4_top10_movies.png`
- `bieu_do_5a_itemcf_recs.png`
- `bieu_do_5b_usercf_recs.png`
- `bieu_do_6_model_comparison.png`
- `bieu_do_7_cold_start.png`
- `bieu_do_8_kfold.png`
- `bieu_do_9_compare_3models.png`
