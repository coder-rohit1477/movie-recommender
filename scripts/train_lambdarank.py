import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import lightgbm as lgb
from typing import List

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.evaluation import precision_at_k, recall_at_k, ndcg_at_k

def train_and_evaluate():
    print("Loading dataset...")
    if not os.path.exists('data/ranking_dataset.csv'):
        print("Error: data/ranking_dataset.csv not found.")
        return
        
    df = pd.read_csv('data/ranking_dataset.csv')
    with open('data/group_sizes.json', 'r') as f:
        groups = json.load(f)

    # 1. Feature Selection
    feature_cols = [
        'svd_score', 'content_score', 'popularity_score', 
        'genre_overlap', 'context_genre_match', 'user_maturity'
    ]
    
    X = df[feature_cols]
    y = df['label']

    # 2. Group-Aware Train/Test Split (80/20 users)
    n_users = len(groups)
    train_user_count = int(0.8 * n_users)
    
    train_row_limit = sum(groups[:train_user_count])
    
    X_train = X.iloc[:train_row_limit]
    y_train = y.iloc[:train_row_limit]
    groups_train = groups[:train_user_count]
    
    X_val = X.iloc[train_row_limit:]
    y_val = y.iloc[train_row_limit:]
    groups_val = groups[train_user_count:]
    
    print(f"Training on {train_user_count} users ({len(X_train)} rows)")
    print(f"Validating on {n_users - train_user_count} users ({len(X_val)} rows)")

    # 3. Model Training
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        label_gain=[0, 1],
        random_state=42,
        importance_type='gain',
        verbose=-1
    )

    print("\nTraining LGBMRanker...")
    ranker.fit(
        X_train, y_train, 
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        eval_at=[10]
    )

    # 4. Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': ranker.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCE (GAIN) ---")
    print(importance.to_string(index=False))

    # 5. Comparative Evaluation
    val_df = df.iloc[train_row_limit:].copy()
    val_df['lgbm_score'] = ranker.predict(X_val)
    
    def phase2c_score(row):
        if row['context_genre_match'] > 0:
            blended = (row['user_maturity'] * row['svd_score']) + ((1 - row['user_maturity']) * row['content_score'])
            blended += (0.1 * row['genre_overlap']) + (0.1 * row['context_genre_match'])
        else:
            blended = row['svd_score'] + (0.1 * row['genre_overlap'])
        return blended + (0.05 * row['popularity_score']) + (0.05 * row['popularity_score'])

    val_df['p2c_score'] = val_df.apply(phase2c_score, axis=1)

    metrics = {'lgbm': {'p':[], 'r':[], 'n':[]}, 'p2c': {'p':[], 'r':[], 'n':[]}}

    print("\nComputing metrics per user group...")
    current_idx = 0
    # Group users by user_id to correctly handle the split rows
    # The groups list corresponds to the sorted order of user_ids in the dataset
    # We verify this alignment:
    user_ids_val = val_df['user_id'].unique()
    
    for i, user_id in enumerate(user_ids_val):
        g_size = groups_val[i]
        user_rows = val_df.iloc[current_idx:current_idx+g_size]
        current_idx += g_size
        
        relevant = user_rows[user_rows['label'] == 1]['title'].tolist()
        if not relevant: continue

        lgbm_recs = user_rows.sort_values('lgbm_score', ascending=False).head(10)['title'].tolist()
        metrics['lgbm']['p'].append(precision_at_k(lgbm_recs, relevant, 10))
        metrics['lgbm']['r'].append(recall_at_k(lgbm_recs, relevant, 10))
        metrics['lgbm']['n'].append(ndcg_at_k(lgbm_recs, relevant, 10))

        p2c_recs = user_rows.sort_values('p2c_score', ascending=False).head(10)['title'].tolist()
        metrics['p2c']['p'].append(precision_at_k(p2c_recs, relevant, 10))
        metrics['p2c']['r'].append(recall_at_k(p2c_recs, relevant, 10))
        metrics['p2c']['n'].append(ndcg_at_k(p2c_recs, relevant, 10))

    print("\n" + "="*70)
    print(f"{'Ranker':<25} | {'Precision@10':<12} | {'Recall@10':<12} | {'NDCG@10':<12}")
    print("-" * 70)
    for model in ['p2c', 'lgbm']:
        name = "Phase 2C (Heuristic)" if model == 'p2c' else "Phase 3 (LambdaRank)"
        print(f"{name:<25} | {np.mean(metrics[model]['p']):<12.4f} | {np.mean(metrics[model]['r']):<12.4f} | {np.mean(metrics[model]['n']):<12.4f}")
    print("="*70)

    os.makedirs('models', exist_ok=True)
    with open('models/lambdarank_model.pkl', 'wb') as f:
        pickle.dump(ranker, f)
    print(f"\nModel saved to models/lambdarank_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
