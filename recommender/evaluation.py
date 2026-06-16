import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)

def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    relevant_set = set(relevant)
    intersection = [r for r in recommended if r in relevant_set]
    return len(intersection) / k

def recall_at_k(recommended, relevant, k):
    if not relevant: return 0.0
    recommended = recommended[:k]
    relevant_set = set(relevant)
    intersection = [r for r in recommended if r in relevant_set]
    return len(intersection) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    if not relevant: return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def create_temporal_split(ratings_df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a per-user temporal split.
    For each user: oldest (1-test_ratio) -> train, newest (test_ratio) -> test.
    """
    train_list = []
    test_list = []
    
    # Ensure timestamp exists
    if 'timestamp' not in ratings_df.columns:
        raise ValueError("ratings_df must contain 'timestamp' for temporal split.")
    
    logger.info("Performing per-user temporal split...")
    for user_id, group in ratings_df.groupby('userId'):
        if len(group) < 5: # Skip users with insufficient history
            continue
            
        group = group.sort_values('timestamp')
        split_idx = int(len(group) * (1 - test_ratio))
        
        if split_idx == 0 or split_idx == len(group):
            continue
            
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
        
    return pd.concat(train_list), pd.concat(test_list)

class Evaluator:
    def __init__(self, k: int = 10, rating_threshold: float = 3.5):
        self.k = k
        self.threshold = rating_threshold
        self.results = {}

    def run(self, model, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Runs evaluation across all users in the test set.
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        # Ground truth: movies user liked in the test set
        relevant_items = test_df[test_df['rating'] >= self.threshold].groupby('userId')['movieId'].apply(list).to_dict()
        
        test_users = list(relevant_items.keys())
        logger.info(f"Evaluating model on {len(test_users)} users...")
        
        for user_id in test_users:
            # 1. Generate Recommendations
            # model.predict returns {title: score}
            recommendations_dict = model.predict(user_id, top_n=self.k)
            recommended_titles = list(recommendations_dict.keys())
            
            # 2. Get ground truth titles
            user_relevant_ids = relevant_items[user_id]
            relevant_titles = [model.movie_titles.get(mid) for mid in user_relevant_ids if mid in model.movie_titles]
            
            if not relevant_titles:
                continue
                
            # 3. Compute Metrics
            precisions.append(precision_at_k(recommended_titles, relevant_titles, self.k))
            recalls.append(recall_at_k(recommended_titles, relevant_titles, self.k))
            ndcgs.append(ndcg_at_k(recommended_titles, relevant_titles, self.k))
            
        self.results = {
            f"Precision@{self.k}": np.mean(precisions) if precisions else 0,
            f"Recall@{self.k}": np.mean(recalls) if recalls else 0,
            f"NDCG@{self.k}": np.mean(ndcgs) if ndcgs else 0
        }
        return self.results

    def report(self):
        print("\n" + "="*30)
        print("EVALUATION REPORT")
        print("="*30)
        for metric, value in self.results.items():
            print(f"{metric:15}: {value:.4f}")
        print("="*30 + "\n")
