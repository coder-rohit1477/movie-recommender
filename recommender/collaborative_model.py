import os
import pickle
from app.data_loader import load_ratings, load_movies, get_surprise_dataset
from app.core.config import settings

class CollaborativeModel:
    def __init__(self):
        self.model = None
        self.trainset = None
        self.movie_titles = None
        self.model_path = os.path.join(settings.MODELS_DIR, "collaborative_svd.pkl")

    def train(self):
        print("Training Collaborative Model (SVD)...")
        ratings = load_ratings()
        movies = load_movies()
        
        if ratings.empty or movies.empty: return

        # Mappings
        self.movie_titles = dict(zip(movies['movieId'], movies['title']))

        # Load into Surprise
        dataset = get_surprise_dataset(ratings)
        
        # Build Trainset
        self.trainset = dataset.build_full_trainset()
        
        # Train SVD
        from surprise import SVD
        self.model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.model.fit(self.trainset)
        
        # Save
        if not os.path.exists(settings.MODELS_DIR): os.makedirs(settings.MODELS_DIR)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'trainset': self.trainset,
                'movie_titles': self.movie_titles
            }, f)
        print("Collaborative Model Saved.")

    def load(self):
        if not os.path.exists(self.model_path):
            self.train()
        else:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.trainset = data['trainset']
                self.movie_titles = data['movie_titles']

    def recommend(self, user_id: str, top_k=10):
        if self.model is None: self.load()
        if self.trainset is None: return [] # Failed to load/train

        user_id = str(user_id)
        # Check if user is known
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            user_rated_items = set(item_id for (item_id, rating) in self.trainset.ur[inner_uid])
        except ValueError:
            # Unknown user -> empty list (Hybrid will handle fallback)
            return []

        candidates = []
        all_items = self.trainset.all_items()
        
        for inner_iid in all_items:
            if inner_iid in user_rated_items: continue
            
            raw_iid = self.trainset.to_raw_iid(inner_iid)
            pred = self.model.predict(user_id, raw_iid)
            candidates.append((raw_iid, pred.est))
        
        # Sort
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for movie_id, score in candidates[:top_k]:
            title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
            results.append({
                "movieId": movie_id,
                "title": str(title),
                "score": float(score),
                "type": "collaborative"
            })
            
        return results

collaborative_model = CollaborativeModel()
