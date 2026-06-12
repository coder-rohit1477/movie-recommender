import os

class Config:
    PROJECT_NAME = "Movie Recommender API"
    VERSION = "2.0.0"
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Dataset files
    RATINGS_FILE_100K = os.path.join(DATA_DIR, "u.data")
    ITEMS_FILE_100K = os.path.join(DATA_DIR, "u.item")
    RATINGS_FILE_20M = os.path.join(DATA_DIR, "ratings.csv")
    MOVIES_FILE_20M = os.path.join(DATA_DIR, "movies.csv")
    
    # Model files
    SVD_MODEL_FILE = os.path.join(MODELS_DIR, "svd_model.pkl")
    CONTENT_MODEL_FILE = os.path.join(MODELS_DIR, "content_model.pkl")
    FAISS_INDEX_FILE = os.path.join(MODELS_DIR, "content_faiss.index")
    
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

settings = Config()
