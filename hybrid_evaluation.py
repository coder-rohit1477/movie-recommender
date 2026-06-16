import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split, Evaluator
from recommender.svd_model import svd_model
from recommender.hybrid_model import hybrid_model
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

class HybridEvaluatorWrapper:
    def __init__(self, model, train_df, movies_df):
        self.model = model
        self.model.ratings_df = train_df
        self.model.movies_df = movies_df
        self.model.compute_popularity_scores()
        self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))

    def predict(self, user_id: str, top_n: int = 10) -> Dict[str, float]:
        recs_list = self.model.recommend(user_id=user_id, top_k=top_n)
        return {item['title']: item['score'] for item in recs_list}

def run_hybrid_comparison():
    print("Loading data...")
    ratings = DataLoader.load_ratings()
    movies = load_movies()
    
    print("Splitting data...")
    train, test = create_temporal_split(ratings)
    
    relevant_users = test[test['rating'] >= 3.5]['userId'].unique()
    np.random.seed(42)
    eval_users = np.random.choice(relevant_users, size=min(200, len(relevant_users)), replace=False)
    test_subset = test[test['userId'].isin(eval_users)]

    # 1. Popularity Baseline
    print("\n[1/3] Evaluating Popularity Baseline...")
    pop_model = PopularityBaseline(train, movies)
    evaluator = Evaluator(k=10)
    pop_results = evaluator.run(pop_model, train, test_subset)
    
    # 2. SVD Model
    print("[2/3] Evaluating SVD Model...")
    svd_model.fit(train, movies)
    svd_results = evaluator.run(svd_model, train, test_subset)
    
    # 3. Hybrid Model
    print("[3/3] Evaluating Hybrid Model...")
    hybrid_wrapper = HybridEvaluatorWrapper(hybrid_model, train, movies)
    hybrid_results = evaluator.run(hybrid_wrapper, train, test_subset)
    
    print("\n" + "="*70)
    print(f"{'Model':<20} | {'Precision@10':<12} | {'Recall@10':<12} | {'NDCG@10':<12}")
    print("-" * 70)
    for name, res in [("Popularity", pop_results), ("SVD", svd_results), ("Hybrid", hybrid_results)]:
        print(f"{name:<20} | {res['Precision@10']:<12.4f} | {res['Recall@10']:<12.4f} | {res['NDCG@10']:<12.4f}")
    print("="*70)
    
    all_res = {"Popularity": pop_results, "SVD": svd_results, "Hybrid": hybrid_results}
    winner = max(all_res, key=lambda x: all_res[x]['Precision@10'])
    base_p = pop_results['Precision@10']
    win_p = all_res[winner]['Precision@10']
    improvement = ((win_p - base_p) / base_p) * 100 if base_p > 0 else 0
    
    print(f"\nWinning Model: {winner}")
    print(f"Improvement over Popularity: {improvement:+.2f}%")

if __name__ == "__main__":
    run_hybrid_comparison()
