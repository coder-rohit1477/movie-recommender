from pathlib import Path
import pandas as pd
from functools import lru_cache
import logging
from surprise import Reader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Robust path handling using pathlib
# This assumes data_loader.py is located in [root]/app/data_loader.py
# BASE_DIR will point to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SAMPLE_SIZE = 3_000_000
RANDOM_STATE = 42

@lru_cache(maxsize=1)
def load_movies():
    """
    Loads and caches the movies dataset from data/movies.csv.
    Returns:
        pd.DataFrame: DataFrame containing movieId and title.
    """
    movies_path = DATA_DIR / "movies.csv"
    
    if not movies_path.exists():
        error_msg = f"Movies dataset not found at: {movies_path.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"Loading movies from {movies_path}")
    df = pd.read_csv(movies_path)
    
    # Ensure movieId is string for consistent lookups and strip titles
    df['movieId'] = df['movieId'].astype(str)
    if 'title' in df.columns:
        df['title'] = df['title'].str.strip()
        
    return df

@lru_cache(maxsize=1)
def load_ratings():
    """
    Loads and caches the ratings dataset from data/ratings.csv.
    Returns:
        pd.DataFrame: DataFrame containing userId, movieId, rating, and timestamp.
    """
    ratings_path = DATA_DIR / "ratings.csv"
    
    if not ratings_path.exists():
        error_msg = f"Ratings dataset not found at: {ratings_path.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"Loading ratings from {ratings_path}")
    
    # Load userId, movieId, rating, and timestamp
    df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'timestamp'])
    
    # User-aware sampling to preserve history
    if len(df) > SAMPLE_SIZE:
        logger.info(f"Applying user-aware sampling (target: approx {SAMPLE_SIZE} ratings)")
        unique_users = df['userId'].unique()
        avg_ratings_per_user = len(df) / len(unique_users)
        n_users_to_sample = int(SAMPLE_SIZE / avg_ratings_per_user)
        
        sampled_users = pd.Series(unique_users).sample(
            n=min(len(unique_users), n_users_to_sample), 
            random_state=RANDOM_STATE
        )
        df = df[df['userId'].isin(sampled_users)]
        logger.info(f"Sampled {len(sampled_users)} users, resulting in {len(df)} ratings")
        
    # Ensure IDs are strings
    df['userId'] = df['userId'].astype(str)
    df['movieId'] = df['movieId'].astype(str)
    
    return df

def get_surprise_dataset(ratings_df):
    """
    Converts a pandas DataFrame into a Surprise Dataset.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
