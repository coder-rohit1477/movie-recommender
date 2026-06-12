import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from app.data_loader import load_movies
from app.core.config import settings

class ContentModel:
    def __init__(self):
        self.index = None
        self.vectorizer = None
        self.movie_ids = None
        self.titles = None
        self.metadata_path = os.path.join(settings.MODELS_DIR, "content_meta.pkl")
        self.index_path = os.path.join(settings.MODELS_DIR, "content.index")

    def train(self):
        print("Training Content Model (TF-IDF + FAISS)...")
        df = load_movies()
        if df.empty: return

        # Feature Engineering: Combine title and genres
        if 'genres' in df.columns:
            df['content'] = df['title'] + " " + df['genres'].str.replace('|', ' ', regex=False)
        else:
            df['content'] = df['title']
            
        # Vectorize
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = self.vectorizer.fit_transform(df['content'])
        
        # Convert to float32 dense matrix for FAISS
        dense_vecs = tfidf_matrix.toarray().astype('float32')
        
        # Normalize (L2) so that Inner Product == Cosine Similarity
        faiss.normalize_L2(dense_vecs)
        
        # Build Index
        d = dense_vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(dense_vecs)
        
        # Save Metadata
        self.movie_ids = df['movieId'].values
        self.titles = df['title'].values
        
        if not os.path.exists(settings.MODELS_DIR): os.makedirs(settings.MODELS_DIR)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer, 
                'movie_ids': self.movie_ids, 
                'titles': self.titles
            }, f)
        print("Content Model Saved.")

    def load(self):
        if not os.path.exists(self.index_path):
            self.train()
        else:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.movie_ids = data['movie_ids']
                self.titles = data['titles']

    def recommend(self, movie_title, top_k=10):
        if self.index is None: self.load()
        if self.titles is None: return []

        # Find movie index
        # Fast Exact Match (Case Insensitive)
        matches = np.where(np.char.lower(self.titles.astype(str)) == movie_title.lower())[0]
        if len(matches) == 0: return []
        
        idx = matches[0]
        
        # Retrieve vector & Search
        vec = self.index.reconstruct(int(idx)).reshape(1, -1)
        scores, indices = self.index.search(vec, top_k + 1)
        
        results = []
        for i, neighbor_idx in enumerate(indices[0]):
            if neighbor_idx == idx: continue # Skip itself
            results.append({
                "movieId": int(self.movie_ids[neighbor_idx]),
                "title": str(self.titles[neighbor_idx]),
                "score": float(scores[0][i]),
                "type": "content"
            })
            
        return results[:top_k]

content_model = ContentModel()
