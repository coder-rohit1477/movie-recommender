import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.data_loader import load_movies
from app.core.config import settings

class FaissIndex:
    def __init__(self):
        self.index = None
        self.df = None
        self.model = None

    def train(self):
        print("Building Semantic FAISS index (all-MiniLM-L6-v2)...")
        movies_df = load_movies()
        
        # Prepare content features
        if 'genres' in movies_df.columns:
            movies_df['genre_str'] = movies_df['genres'].str.replace('|', ' ', regex=False)
        else:
            movies_df['genre_str'] = ""
            
        content_features = (movies_df['title'] + " " + movies_df['genre_str']).tolist()
        self.df = movies_df[['movieId', 'title']].copy()
        
        # Use SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self.model.encode(content_features, show_progress_bar=True)
        
        # Convert to float32 and Normalize for Cosine Similarity (Inner Product)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Build Inner Product index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(np.ascontiguousarray(embeddings))
        
        if not os.path.exists(settings.MODELS_DIR):
            os.makedirs(settings.MODELS_DIR)
            
        faiss.write_index(self.index, settings.FAISS_INDEX_FILE)
        with open(settings.CONTENT_MODEL_FILE, 'wb') as f:
            pickle.dump({
                'df': self.df
            }, f)
        print("Semantic FAISS index built and saved.")

    def load(self):
        if not os.path.exists(settings.FAISS_INDEX_FILE) or not os.path.exists(settings.CONTENT_MODEL_FILE):
            self.train()
        else:
            self.index = faiss.read_index(settings.FAISS_INDEX_FILE)
            with open(settings.CONTENT_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.df = data['df']
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Semantic FAISS model loaded from disk.")

    def search(self, movie_title, top_n=10):
        if self.model is None:
            self.load()
            
        movie_title = str(movie_title).strip()
        match = self.df[self.df['title'].str.lower() == movie_title.lower()]
        
        if match.empty:
            return {}

        # Encode query title
        query_vec = self.model.encode([movie_title]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search (Scores are Cosine Similarities)
        scores, indices = self.index.search(query_vec, top_n + 1)
        
        results = {}
        for i, neighbor_idx in enumerate(indices[0]):
            title = self.df.iloc[neighbor_idx]['title']
            if title.lower() == movie_title.lower():
                continue
            
            results[title] = float(scores[0][i])
            if len(results) >= top_n:
                break
                
        return results

faiss_index = FaissIndex()
