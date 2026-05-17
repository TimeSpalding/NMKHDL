import os

# Đường dẫn
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_DIR   = os.path.join(BASE_DIR, "data")        
REPORT_DIR = os.path.join(BASE_DIR, "report")   

RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH  = os.path.join(DATA_DIR, "movies.csv")
TAGS_PATH    = os.path.join(DATA_DIR, "tags.csv")
LINKS_PATH   = os.path.join(DATA_DIR, "links.csv")

# Tham số mô hình
SAMPLE_SIZE   = 5000     # số ratings lấy mẫu
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
MIN_USER_RATINGS  = 5    # lọc cold-start users
N_NEIGHBORS   = 10       # số neighbors dùng trong CF
SVD_FACTORS   = 20       # số latent factors cho SVD
KFOLD_SPLITS  = 5

# Ranking Metrics
RELEVANCE_THRESHOLD = 4.0        # rating >= này mới coi là "relevant"
TOP_K_LIST          = [5, 10, 20]  # các giá trị K để đánh giá
