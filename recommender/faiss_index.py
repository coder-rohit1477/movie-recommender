import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from app.data_loader import load_movies
from app.core.config import settings

class FaissIndex:
    def __init__(self):
        self.index = None
        self.df = None
        self.vectorizer = None

    def train(self):
        print("Building FAISS index...")
        movies_df = load_movies()
        
        # Prepare content features
        if 'genres' in movies_df.columns:
            movies_df['genre_str'] = movies_df['genres'].str.replace('|', ' ', regex=False)
        else:
            movies_df['genre_str'] = ""
            
        movies_df['content_features'] = movies_df['title'] + " " + movies_df['genre_str']
        self.df = movies_df[['movieId', 'title']].copy()
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(movies_df['content_features'])
        
        # Convert TF-IDF to dense float32 for FAISS
        dense_matrix = tfidf_matrix.toarray().astype('float32')
        
        # Build L2 index
        d = dense_matrix.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(dense_vecs := np.ascontiguousarray(dense_matrix))
        
        if not os.path.exists(settings.MODELS_DIR):
            os.makedirs(settings.MODELS_DIR)
            
        faiss.write_index(self.index, settings.FAISS_INDEX_FILE)
        with open(settings.CONTENT_MODEL_FILE, 'wb') as f:
            pickle.dump({
                'df': self.df,
                'vectorizer': self.vectorizer
            }, f)
        print("FAISS index built and saved.")

    def load(self):
        if not os.path.exists(settings.FAISS_INDEX_FILE) or not os.path.exists(settings.CONTENT_MODEL_FILE):
            self.train()
        else:
            self.index = faiss.read_index(settings.FAISS_INDEX_FILE)
            with open(settings.CONTENT_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.df = data['df']
                self.vectorizer = data['vectorizer']
            print("FAISS index loaded from disk.")

    def search(self, movie_title, top_n=10):
        movie_title = str(movie_title).strip()
        match = self.df[self.df['title'].str.lower() == movie_title.lower()]
        
        if match.empty:
            return {}

        idx = match.index[0]
        
        # Get the vector for the movie
        # Since we use IndexFlatL2, we need the original vector. 
        # Re-vectorizing or storing? Storing dense matrix is memory heavy.
        # Let's re-vectorize the movie's title + genre (this is fast for 1 item)
        # Wait, if we have many movies, we can't re-vectorize all.
        # But we only need the vector for ONE movie to search for its neighbors.
        
        # Actually, let's just use the index reconstruct if possible, 
        # but IndexFlatL2 does not always support it. 
        # Re-vectorizing the content feature of the matched movie:
        
        # We need the original content_features to re-vectorize.
        # Let's include it in the pickle.
        
        # Optimization: We already have the vectorizer. We just need the content string.
        # I'll update train to include content_features in df.
        
        # Search:
        # Re-fetching features for matching movie is slow if we don't store them.
        # Let's assume we have it.
        # Actually, for now, let's use the index position.
        # FAISS IndexFlatL2 preserves order.
        
        query_vector = self.index.reconstruct(int(idx)).reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, top_n + 1)
        
        results = {}
        for i, neighbor_idx in enumerate(indices[0]):
            if neighbor_idx == idx:
                continue
            title = self.df.iloc[neighbor_idx]['title']
            # Distance to Similarity conversion (heuristic)
            score = 1.0 / (1.0 + distances[0][i])
            results[title] = float(score)
            
        return results

faiss_index = FaissIndex()
