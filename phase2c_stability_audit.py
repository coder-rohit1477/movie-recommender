import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split, Evaluator
from recommender.svd_model import svd_model
from recommender.hybrid_model import hybrid_model
from recommender.faiss_index import faiss_index
from app.data_loader import load_movies
from true_hybrid_evaluation import ContextAwareHybridWrapper, PopularityBaseline

def run_audit():
    print("Loading data...")
    ratings = DataLoader.load_ratings()
    movies = load_movies()
    
    print("Preparing Split...")
    train, test = create_temporal_split(ratings)
    
    print("Fitting SVD...")
    svd_model.fit(train, movies)
    faiss_index.load()
    
    relevant_users = test[test['rating'] >= 3.5]['userId'].unique()
    
    results_table = []
    
    for count in [50, 100, 200]:
        print(f"\n--- Auditing {count} Users ---")
        np.random.seed(42)
        eval_users = np.random.choice(relevant_users, size=min(count, len(relevant_users)), replace=False)
        test_subset = test[test['userId'].isin(eval_users)]
        
        evaluator = Evaluator(k=10)
        
        # SVD
        svd_res = evaluator.run(svd_model, train, test_subset)
        # Pop
        pop_model = PopularityBaseline(train, movies)
        pop_res = evaluator.run(pop_model, train, test_subset)
        # Hybrid (Phase 2C)
        wrapper = ContextAwareHybridWrapper(hybrid_model, train, movies)
        hybrid_res = evaluator.run(wrapper, train, test_subset)
        
        results_table.append({
            "users": count,
            "SVD_P10": svd_res['Precision@10'],
            "Pop_P10": pop_res['Precision@10'],
            "Hybrid_P10": hybrid_res['Precision@10'],
            "Hybrid_R10": hybrid_res['Recall@10'],
            "Hybrid_NDCG10": hybrid_res['NDCG@10']
        })

    print("\n" + "="*80)
    print(f"{'Users':<6} | {'SVD P@10':<8} | {'Pop P@10':<8} | {'Hybrid P@10':<11} | {'Hybrid R@10':<11} | {'Hybrid NDCG'}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['users']:<6} | {r['SVD_P10']:<8.4f} | {r['Pop_P10']:<8.4f} | {r['Hybrid_P10']:<11.4f} | {r['Hybrid_R10']:<11.4f} | {r['Hybrid_NDCG10']:.4f}")
    print("="*80)

if __name__ == "__main__":
    run_audit()
