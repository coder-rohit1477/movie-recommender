import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
RATINGS_FILE_100K = os.path.join(DATA_DIR, "u.data")
ITEMS_FILE_100K = os.path.join(DATA_DIR, "u.item")
RATINGS_FILE_20M = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_FILE_20M = os.path.join(DATA_DIR, "movies.csv")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
