import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split, Evaluator
from recommender.svd_model import svd_model
from recommender.hybrid_model import hybrid_model
from recommender.faiss_index import faiss_index
from app.data_loader import load_movies

class PopularityBaseline:
    def __init__(self, train_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
        self.train_df = train_df
        stats = train_df.groupby('movieId').agg({'rating': ['count', 'mean']})
        stats.columns = ['vote_count', 'vote_average']
        C = stats['vote_average'].mean()
        m = stats['vote_count'].quantile(0.90)
        qualified = stats[stats['vote_count'] >= m].copy()
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v+m) * R) + (m / (v+m) * C)
        qualified['score'] = qualified.apply(weighted_rating, axis=1)
        self.popular_list = qualified.sort_values('score', ascending=False)

    def predict(self, user_id: str, top_n: int = 10) -> Dict[str, float]:
        user_id = str(user_id)
        user_history = set(self.train_df[self.train_df['userId'] == user_id]['movieId'])
        recs = {}
        for mid, row in self.popular_list.iterrows():
            mid_str = str(mid)
            if mid_str not in user_history:
                title = self.movie_titles.get(mid_str)
                if title: recs[title] = float(row['score'])
                if len(recs) >= top_n: break
        return recs

class ContextAwareHybridWrapper:
    """
    Wraps HybridModel to provide context-aware recommendations.
    Retrieves the most recent liked movie from training data as context.
    """
    def __init__(self, model, train_df, movies_df):
        self.model = model
        self.train_df = train_df
        self.movies_df = movies_df
        self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
        
        # Prepare HybridModel internal state
        self.model.ratings_df = train_df
        self.model.movies_df = movies_df
        self.model.compute_popularity_scores()
        self.model._movie_titles_set = set(movies_df['title'].str.lower().unique())
        
        self.pool_sizes = []
        self.ret_latencies = []

    def _get_context_movie(self, user_id: str) -> Optional[str]:
        user_history = self.train_df[self.train_df['userId'] == str(user_id)]
        liked_movies = user_history[user_history['rating'] >= 3.5]
        
        if liked_movies.empty:
            return None
            
        # Get most recent
        latest_movie_id = liked_movies.sort_values('timestamp', ascending=False).iloc[0]['movieId']
        return self.movie_titles.get(str(latest_movie_id))

    def predict(self, user_id: str, top_n: int = 10) -> Dict[str, float]:
        import time
        context_title = self._get_context_movie(user_id)
        
        # Stage 1 Retrieval (Tracked manually here for audit)
        start_ret = time.time()
        candidates_data = self.model._retrieve_candidates(user_id, context_title, n_svd=100, n_content=100, n_pop=50)
        self.ret_latencies.append(time.time() - start_ret)
        self.pool_sizes.append(len(candidates_data["titles"]))
        
        # Stage 2 Ranking
        recs_list = self.model._rank_candidates(user_id, candidates_data, movie_title=context_title, top_k=top_n)
        return {item['title']: item['score'] for item in recs_list}

def run_true_hybrid_evaluation():
    # ... rest of function remains the same ...
    # After hybrid_results = evaluator.run(...) add:
    # print(f"Avg Candidate Pool Size: {np.mean(hybrid_wrapper.pool_sizes):.1f}")
    # print(f"Avg Retrieval Latency:  {np.mean(hybrid_wrapper.ret_latencies)*1000:.2f} ms")
    print("Loading data...")
    ratings = DataLoader.load_ratings()
    movies = load_movies()
    
    print("Preparing Models...")
    train, test = create_temporal_split(ratings)
    
    # 1. Prepare SVD (Global instance used by Hybrid)
    svd_model.fit(train, movies)
    
    # 2. Prepare FAISS (Needed for Content)
    faiss_index.load()
    
    # Selection of Users
    relevant_users = test[test['rating'] >= 3.5]['userId'].unique()
    np.random.seed(42)
    eval_users = np.random.choice(relevant_users, size=min(50, len(relevant_users)), replace=False)
    test_subset = test[test['userId'].isin(eval_users)]

    # Evaluation Harness
    evaluator = Evaluator(k=10)

    # 1. Popularity Baseline
    print("\n[1/3] Evaluating Popularity Baseline...")
    pop_model = PopularityBaseline(train, movies)
    pop_results = evaluator.run(pop_model, train, test_subset)
    
    # 2. SVD Model
    print("[2/3] Evaluating SVD Model...")
    svd_results = evaluator.run(svd_model, train, test_subset)
    
    # 3. Context-Aware Hybrid
    print("[3/3] Evaluating Context-Aware Hybrid...")
    hybrid_wrapper = ContextAwareHybridWrapper(hybrid_model, train, movies)
    hybrid_results = evaluator.run(hybrid_wrapper, train, test_subset)
    
    # Final Table
    print("\n" + "="*70)
    print(f"{'Model':<20} | {'Precision@10':<12} | {'Recall@10':<12} | {'NDCG@10':<12}")
    print("-" * 70)
    for name, res in [("Popularity", pop_results), ("SVD", svd_results), ("True Hybrid", hybrid_results)]:
        print(f"{name:<20} | {res['Precision@10']:<12.4f} | {res['Recall@10']:<12.4f} | {res['NDCG@10']:<12.4f}")
    print("="*70)
    
    print(f"\nPhase 2B Audit Stats:")
    print(f"Avg Candidate Pool Size: {np.mean(hybrid_wrapper.pool_sizes):.1f}")
    print(f"Avg Retrieval Latency:  {np.mean(hybrid_wrapper.ret_latencies)*1000:.2f} ms")

if __name__ == "__main__":
    run_true_hybrid_evaluation()
