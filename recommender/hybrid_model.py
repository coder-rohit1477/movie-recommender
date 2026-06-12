from recommender.svd_model import svd_model
from recommender.faiss_index import faiss_index
from app.data_loader import load_movies, load_ratings
import pandas as pd
import logging
from typing import List, Dict, Set

# Configure logger
logger = logging.getLogger(__name__)

class HybridModel:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.popularity_scores = {}
        self._movie_titles_set = set()

    def load_metadata(self):
        """Pre-loads movies and ratings into memory for fast lookup."""
        try:
            logger.info("Loading metadata for Hybrid Model...")
            self.movies_df = load_movies()
            self.ratings_df = load_ratings()
            self._movie_titles_set = set(self.movies_df['title'].str.lower().unique())
            self.compute_popularity_scores()
            logger.info("Metadata loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def validate_movie(self, title: str) -> bool:
        """Checks if a movie title exists in the database (case-insensitive)."""
        if not title:
            return False
        return title.lower() in self._movie_titles_set

    def compute_popularity_scores(self):
        """Computes normalized popularity scores for all movies."""
        if self.ratings_df is None or self.movies_df is None:
            return

        counts = self.ratings_df['movieId'].value_counts()
        titles = dict(zip(self.movies_df['movieId'], self.movies_df['title']))
        
        pop_scores = {titles.get(mid): count for mid, count in counts.items() if mid in titles}
        
        if not pop_scores:
            self.popularity_scores = {}
            return

        min_s = min(pop_scores.values())
        max_s = max(pop_scores.values())
        
        if max_s == min_s:
            self.popularity_scores = {k: 1.0 for k in pop_scores}
        else:
            self.popularity_scores = {k: (v - min_s) / (max_s - min_s) for k, v in pop_scores.items()}

    def get_rated_movies(self, user_id: str) -> Set[str]:
        """Returns the set of movie titles already rated by a user."""
        if self.ratings_df is None or self.movies_df is None:
            return set()
            
        user_ratings = self.ratings_df[self.ratings_df['userId'] == str(user_id)]
        if user_ratings.empty:
            return set()
            
        return set(self.movies_df[self.movies_df['movieId'].isin(user_ratings['movieId'])]['title'])

    def recommend(self, user_id: str, movie_title: str = None, top_k: int = 10) -> List[Dict]:
        """
        Combines SVD and FAISS with popularity weighting.
        Handles unknown users by defaulting to collaborative filtering on all items or popularity.
        """
        try:
            user_id = str(user_id)
            
            # 1. Collaborative Recommendations (SVD)
            # Handles unknown users internally by returning empty/default scores
            cf_recs = svd_model.predict(user_id, top_n=top_k * 5)
            # Normalize CF scores (SVD 1-5 -> 0-1)
            cf_normalized = {k: (v - 1.0) / 4.0 for k, v in cf_recs.items()}
            
            # 2. Content-based Recommendations (FAISS)
            cb_normalized = {}
            if movie_title:
                cb_recs = faiss_index.search(movie_title, top_n=top_k * 5)
                # FAISS scores are already 0-1 similarity
                cb_normalized = cb_recs

            # 3. Blending Logic
            rated_movies = self.get_rated_movies(user_id)
            candidate_titles = set(cf_normalized.keys()) | set(cb_normalized.keys())
            
            # If both are empty (rare), fallback to popularity
            if not candidate_titles:
                logger.info(f"No candidates found for user {user_id}. Using popularity fallback.")
                sorted_pop = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
                return [{"title": t, "score": round(float(s), 4)} for t, s in sorted_pop[:top_k]]

            # Weighted Alpha based on user maturity
            rating_count = len(self.ratings_df[self.ratings_df['userId'] == user_id]) if self.ratings_df is not None else 0
            if rating_count < 10: alpha = 0.3 # New user: prefer content
            elif rating_count <= 50: alpha = 0.6
            else: alpha = 0.8 # Mature user: prefer collaborative

            final_scores = []
            for title in candidate_titles:
                if title in rated_movies:
                    continue
                
                cf_score = cf_normalized.get(title, 0.0)
                cb_score = cb_normalized.get(title, 0.0)
                pop_score = self.popularity_scores.get(title, 0.0)
                
                if movie_title:
                    blended = (alpha * cf_score) + ((1 - alpha) * cb_score)
                else:
                    blended = cf_score
                
                # Small boost from popularity (5%)
                final_score = blended + (0.05 * pop_score)
                final_scores.append({
                    "title": title, 
                    "score": round(min(1.0, float(final_score)), 4)
                })

            final_scores.sort(key=lambda x: x['score'], reverse=True)
            return final_scores[:top_k]

        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            raise

hybrid_model = HybridModel()
