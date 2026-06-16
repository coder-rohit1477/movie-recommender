from recommender.svd_model import svd_model
from recommender.faiss_index import faiss_index
from recommender.ranker import ranker
from app.data_loader import load_movies, load_ratings
import pandas as pd
import logging
import time
from typing import List, Dict, Set

# Configure logger
logger = logging.getLogger(__name__)

class HybridModel:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.popularity_scores = {}
        self._movie_titles_set = set()
        self.USE_LTR_RANKER = True # Feature Flag
        self.ranker_loaded = False

    def load_metadata(self):
        """Pre-loads movies and ratings into memory for fast lookup."""
        try:
            logger.info("Loading metadata for Hybrid Model...")
            self.movies_df = load_movies()
            self.ratings_df = load_ratings()
            self._movie_titles_set = set(self.movies_df['title'].str.lower().unique())
            self.compute_popularity_scores()
            
            # Load LTR Ranker
            self.ranker_loaded = ranker.load()
            
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

    def _retrieve_candidates(self, user_id: str, movie_title: str = None, n_svd: int = 100, n_content: int = 100, n_pop: int = 50) -> Dict:
        """
        Stage 1: Candidate Retrieval (High Recall).
        Aggregates candidates from SVD, FAISS, and Global Popularity.
        """
        # 1. Collaborative Retriever (SVD)
        cf_recs = svd_model.predict(user_id, top_n=n_svd)
        cf_scores = {k: (v - 1.0) / 4.0 for k, v in cf_recs.items()}
        
        # 2. Content Retriever (FAISS)
        cb_scores = {}
        if movie_title:
            cb_recs = faiss_index.search(movie_title, top_n=n_content)
            cb_scores = cb_recs

        # 3. Popularity Retriever (Global Trends)
        sorted_pop = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        pop_candidates = {t for t, _ in sorted_pop[:n_pop]}

        # Merge and Deduplicate
        candidate_titles = set(cf_scores.keys()) | set(cb_scores.keys()) | pop_candidates
        
        return {
            "titles": candidate_titles,
            "cf_scores": cf_scores,
            "cb_scores": cb_scores
        }

    def _get_candidate_features(self, user_id: str, candidate_titles: Set[str], cf_scores: Dict, cb_scores: Dict, movie_title: str = None) -> pd.DataFrame:
        """
        Extracts numerical features for a list of candidates.
        Used by both the internal ranker and external dataset generators.
        """
        # 1. Get User Genre Preferences (Top 3)
        user_history = self.ratings_df[self.ratings_df['userId'] == user_id] if self.ratings_df is not None else pd.DataFrame()
        liked_mids = user_history[user_history['rating'] >= 3.5]['movieId']
        user_genres_series = self.movies_df[self.movies_df['movieId'].isin(liked_mids)]['genres'].str.split('|').explode()
        top_user_genres = set(user_genres_series.value_counts().head(3).index) if not user_genres_series.empty else set()

        # 2. Get Context Movie Genres
        context_genres = set()
        if movie_title:
            match = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()]
            if not match.empty:
                context_genres = set(match.iloc[0]['genres'].split('|'))

        # 3. Global Max Votes for normalization
        vote_counts = self.ratings_df['movieId'].value_counts() if self.ratings_df is not None else pd.Series()
        max_votes = vote_counts.max() if not vote_counts.empty else 1

        # 4. Maturity-based Alpha
        rating_count = len(user_history)
        if rating_count < 10: alpha = 0.3
        elif rating_count <= 50: alpha = 0.6
        else: alpha = 0.8

        final_rows = []
        # Pre-filter movies_df for faster lookup
        candidates_df = self.movies_df[self.movies_df['title'].isin(candidate_titles)].drop_duplicates('title').set_index('title')

        for title in candidate_titles:
            if title not in candidates_df.index:
                continue

            movie_info = candidates_df.loc[title]
            m_genres = set(movie_info['genres'].split('|'))
            
            # Feature calculation
            genre_overlap = len(m_genres & top_user_genres) / 3.0 if top_user_genres else 0.0
            context_match = 1.0 if context_genres and (m_genres & context_genres) else 0.0
            mid = movie_info['movieId']
            vote_conf = vote_counts.get(mid, 0) / max_votes
            
            final_rows.append({
                "user_id": user_id,
                "title": title,
                "svd_score": cf_scores.get(title, 0.0),
                "content_score": cb_scores.get(title, 0.0),
                "popularity_score": self.popularity_scores.get(title, 0.0),
                "genre_overlap": genre_overlap,
                "context_genre_match": context_match,
                "vote_confidence": vote_conf,
                "user_maturity": alpha
            })
            
        return pd.DataFrame(final_rows)

    def _rank_candidates(self, user_id: str, candidates_data: Dict, movie_title: str = None, top_k: int = 10) -> List[Dict]:
        """
        Stage 2: Ranking Layer (High Precision).
        Orchestrates ML-based LambdaRank or heuristic fallback.
        """
        candidate_titles = candidates_data["titles"]
        cf_scores = candidates_data["cf_scores"]
        cb_scores = candidates_data["cb_scores"]
        
        rated_movies = self.get_rated_movies(user_id)
        
        # 1. Extract Features
        features_df = self._get_candidate_features(user_id, candidate_titles, cf_scores, cb_scores, movie_title)
        
        if features_df.empty:
            return []

        # 2. Strategy A: LambdaRank (Preferred)
        if self.USE_LTR_RANKER and self.ranker_loaded:
            try:
                start_inf = time.time()
                # Inference
                scores = ranker.predict(features_df)
                features_df['final_score'] = scores
                latency_ms = (time.time() - start_inf) * 1000
                logger.info(f"LTR Inference Success: {latency_ms:.2f}ms for {len(features_df)} candidates.")

                final_scores = []
                for _, row in features_df.iterrows():
                    if row['title'] in rated_movies: continue
                    final_scores.append({
                        "title": row['title'], 
                        "score": round(float(row['final_score']), 4)
                    })
                
                final_scores.sort(key=lambda x: x['score'], reverse=True)
                return final_scores[:top_k]
            except Exception as e:
                logger.error(f"LambdaRank inference failed, falling back to heuristic: {e}")

        # 3. Strategy B: Heuristic Ranking (Fallback)
        final_scores = []
        for _, row in features_df.iterrows():
            title = row['title']
            if title in rated_movies: continue
            
            alpha = row['user_maturity']
            if movie_title:
                blended = (alpha * row['svd_score']) + ((1 - alpha) * row['content_score'])
                blended += (0.1 * row['genre_overlap']) + (0.1 * row['context_genre_match'])
            else:
                blended = row['svd_score'] + (0.1 * row['genre_overlap'])
            
            final_score = blended + (0.05 * row['popularity_score']) + (0.05 * row['vote_confidence'])
            
            final_scores.append({
                "title": title, 
                "score": round(min(1.0, float(final_score)), 4)
            })

        final_scores.sort(key=lambda x: x['score'], reverse=True)
        return final_scores[:top_k]

    def recommend(self, user_id: str, movie_title: str = None, top_k: int = 10) -> List[Dict]:
        """
        Public API: Orchestrates the two-stage pipeline.
        """
        try:
            user_id = str(user_id)
            
            # Stage 1: Retrieval (Candidate Generation)
            candidates_data = self._retrieve_candidates(user_id, movie_title, n_svd=100, n_content=100, n_pop=50)
            
            # Fallback if no candidates found (rare)
            if not candidates_data["titles"]:
                logger.info(f"No candidates found for user {user_id}. Using popularity fallback.")
                sorted_pop = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
                return [{"title": t, "score": round(float(s), 4)} for t, s in sorted_pop[:top_k]]

            # Stage 2: Ranking
            return self._rank_candidates(user_id, candidates_data, movie_title, top_k)

        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            raise

hybrid_model = HybridModel()
