import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from app.data_loader import load_movies
from app.core.config import settings

class ContentModel:
    def __init__(self):
        self.index = None
        self.model = None
        self.movie_ids = None
        self.titles = None
        self.metadata_path = os.path.join(settings.MODELS_DIR, "content_meta.pkl")
        self.index_path = os.path.join(settings.MODELS_DIR, "content.index")

    def train(self):
        print("Training Semantic Content Model (all-MiniLM-L6-v2)...")
        df = load_movies()
        if df.empty: return

        # Feature Engineering
        if 'genres' in df.columns:
            df['content'] = df['title'] + " " + df['genres'].str.replace('|', ' ', regex=False)
        else:
            df['content'] = df['title']
            
        # Semantic Vectorization
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = self.model.encode(df['content'].tolist(), show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Build Index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(np.ascontiguousarray(embeddings))
        
        # Save Metadata
        self.movie_ids = df['movieId'].values
        self.titles = df['title'].values
        
        if not os.path.exists(settings.MODELS_DIR): os.makedirs(settings.MODELS_DIR)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'movie_ids': self.movie_ids, 
                'titles': self.titles
            }, f)
        print("Semantic Content Model Saved.")

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            self.train()
        else:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.movie_ids = data['movie_ids']
                self.titles = data['titles']
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Semantic Content Model loaded from disk.")

    def recommend(self, movie_title, top_k=10):
        if self.model is None: self.load()
        if self.titles is None: return []

        # Encode query
        query_vec = self.model.encode([movie_title]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search
        scores, indices = self.index.search(query_vec, top_k + 1)
        
        results = []
        for i, neighbor_idx in enumerate(indices[0]):
            title = self.titles[neighbor_idx]
            if title.lower() == movie_title.lower(): continue
            
            results.append({
                "movieId": int(self.movie_ids[neighbor_idx]),
                "title": str(title),
                "score": float(scores[0][i]),
                "type": "content"
            })
            
        return results[:top_k]

content_model = ContentModel()
