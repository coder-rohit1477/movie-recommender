import sys
import os
import pandas as pd
import numpy as np
from typing import Dict

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split, Evaluator
from recommender.svd_model import SVDModel
from app.data_loader import load_movies

class PopularityBaseline:
    """
    Implements the same IMDb Weighted Rating logic as PopularityModel
    but fits strictly on the provided training dataframe.
    """
    def __init__(self, train_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
        self.train_df = train_df
        
        # 1. Compute Popularity Scores
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
        # Exclude already rated movies in train_df
        user_history = set(self.train_df[self.train_df['userId'] == user_id]['movieId'])
        
        recs = {}
        for mid, row in self.popular_list.iterrows():
            mid_str = str(mid)
            if mid_str not in user_history:
                title = self.movie_titles.get(mid_str)
                if title:
                    recs[title] = float(row['score'])
                if len(recs) >= top_n:
                    break
        return recs

def run_comparison():
    print("Loading data...")
    ratings = DataLoader.load_ratings()
    movies = load_movies()
    
    print("Splitting data...")
    train, test = create_temporal_split(ratings)
    
    # Subset for speed
    relevant_users = test[test['rating'] >= 3.5]['userId'].unique()
    np.random.seed(42)
    eval_users = np.random.choice(relevant_users, size=min(200, len(relevant_users)), replace=False)
    test_subset = test[test['userId'].isin(eval_users)]

    # 1. Evaluate Popularity Baseline
    print("\nEvaluating Popularity Baseline...")
    pop_model = PopularityBaseline(train, movies)
    evaluator = Evaluator(k=10)
    pop_results = evaluator.run(pop_model, train, test_subset)
    
    # 2. Evaluate SVD Model
    print("Evaluating SVD Model...")
    svd_model = SVDModel()
    svd_model.fit(train, movies)
    svd_results = evaluator.run(svd_model, train, test_subset)
    
    # 3. Output Table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Precision@10':<12} | {'Recall@10':<12} | {'NDCG@10':<12}")
    print("-" * 60)
    print(f"{'Popularity':<20} | {pop_results['Precision@10']:<12.4f} | {pop_results['Recall@10']:<12.4f} | {pop_results['NDCG@10']:<12.4f}")
    print(f"{'SVD':<20} | {svd_results['Precision@10']:<12.4f} | {svd_results['Recall@10']:<12.4f} | {svd_results['NDCG@10']:<12.4f}")
    print("="*60)
    
    # 4. Calculate Improvement
    base_p = pop_results['Precision@10']
    svd_p = svd_results['Precision@10']
    improvement = ((svd_p - base_p) / base_p) * 100 if base_p > 0 else 0
    
    print(f"\nSVD Precision Improvement over Popularity: {improvement:+.2f}%")

if __name__ == "__main__":
    run_comparison()
