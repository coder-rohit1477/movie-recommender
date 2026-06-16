import pandas as pd
import os
from surprise import Reader, Dataset
from app.core.config import settings

SAMPLE_SIZE = 3_000_000
RANDOM_STATE = 42

class DataLoader:
    @staticmethod
    def load_ratings():
        if os.path.exists(settings.RATINGS_FILE_20M):
            # Load with timestamp
            df = pd.read_csv(settings.RATINGS_FILE_20M, usecols=['userId', 'movieId', 'rating', 'timestamp'])
            
            # User-aware sampling
            if len(df) > SAMPLE_SIZE:
                unique_users = df['userId'].unique()
                avg_ratings_per_user = len(df) / len(unique_users)
                n_users_to_sample = int(SAMPLE_SIZE / avg_ratings_per_user)
                sampled_users = pd.Series(unique_users).sample(
                    n=min(len(unique_users), n_users_to_sample), 
                    random_state=RANDOM_STATE
                )
                df = df[df['userId'].isin(sampled_users)]
                
            df['userId'] = df['userId'].astype(str)
            df['movieId'] = df['movieId'].astype(str)
            return df
        
        if os.path.exists(settings.RATINGS_FILE_100K):
            df = pd.read_csv(settings.RATINGS_FILE_100K, sep='\t', header=None, 
                             names=['userId', 'movieId', 'rating', 'timestamp'])
            df['userId'] = df['userId'].astype(str)
            df['movieId'] = df['movieId'].astype(str)
            return df
        
        raise FileNotFoundError("No ratings data found.")

    @staticmethod
    def load_movies():
        if os.path.exists(settings.MOVIES_FILE_20M):
            df = pd.read_csv(settings.MOVIES_FILE_20M)
            df['movieId'] = df['movieId'].astype(str)
            df['title'] = df['title'].str.strip()
            return df
        
        if os.path.exists(settings.ITEMS_FILE_100K):
            df = pd.read_csv(settings.ITEMS_FILE_100K, sep='|', header=None, encoding='latin-1', 
                             usecols=[0, 1], names=['movieId', 'title'])
            df['movieId'] = df['movieId'].astype(str)
            df['title'] = df['title'].str.strip()
            return df
            
        raise FileNotFoundError("No movies data found.")

    @staticmethod
    def get_surprise_dataset(ratings_df):
        reader = Reader(rating_scale=(0.5, 5.0))
        return Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
