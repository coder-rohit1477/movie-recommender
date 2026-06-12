import os
import pickle
import pandas as pd
from surprise import SVD
from app.data_loader import load_ratings, load_movies, get_surprise_dataset
from app.core.config import settings

class SVDModel:
    def __init__(self):
        self.model = None
        self.movie_titles = None
        self.full_trainset = None
        self.all_movie_ids = None

    def train(self):
        print("Training SVD model...")
        try:
            ratings_df = load_ratings()
            movies_df = load_movies()
            
            dataset = get_surprise_dataset(ratings_df)
            
            # For predict() to work, we need these in memory
            self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
            self.all_movie_ids = list(self.movie_titles.keys())
            
            self.model = SVD(n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.02, random_state=42)
            self.full_trainset = dataset.build_full_trainset()
            self.model.fit(self.full_trainset)
            
            if not os.path.exists(settings.MODELS_DIR):
                os.makedirs(settings.MODELS_DIR)
                
            print("Saving SVD model...")
            with open(settings.SVD_MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'svd_model': self.model, 
                    'full_trainset': self.full_trainset
                }, f)
            print("SVD model trained and saved.")
        except Exception as e:
            print(f"Error during SVD training: {e}")

    def load(self):
        should_train = False
        
        if not os.path.exists(settings.SVD_MODEL_FILE):
            should_train = True
        else:
            try:
                with open(settings.SVD_MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    
                if 'svd_model' in data and 'full_trainset' in data:
                    self.model = data['svd_model']
                    self.full_trainset = data['full_trainset']
                    
                    # Also need movie metadata for recommendations
                    movies_df = load_movies()
                    self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
                    self.all_movie_ids = list(self.movie_titles.keys())
                    
                    print("SVD model loaded successfully")
                else:
                    print("Incompatible model format found. Retraining...")
                    should_train = True
            except Exception as e:
                print(f"Error loading SVD model: {e}. Retraining...")
                should_train = True
        
        if should_train:
            self.train()

    def predict(self, user_id, top_n=10):
        if self.model is None or self.full_trainset is None or self.all_movie_ids is None:
            print("Model not loaded. Cannot predict.")
            return {}
            
        user_id = str(user_id)
        
        try:
            inner_user_id = self.full_trainset.to_inner_uid(user_id)
            rated_items = {self.full_trainset.to_raw_iid(i) for (i, _) in self.full_trainset.ur[inner_user_id]}
        except ValueError:
            rated_items = set()

        predictions = []
        for movie_id in self.all_movie_ids:
            if movie_id not in rated_items:
                pred = self.model.predict(user_id, movie_id)
                predictions.append((movie_id, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return {self.movie_titles.get(mid, "Unknown"): float(est) for mid, est in predictions[:top_n]}

svd_model = SVDModel()
