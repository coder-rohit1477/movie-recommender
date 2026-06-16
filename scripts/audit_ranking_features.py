import pandas as pd
import numpy as np

def audit():
    if not os.path.exists('data/ranking_dataset.csv'):
        print("Error: data/ranking_dataset.csv not found.")
        return

    df = pd.read_csv('data/ranking_dataset.csv')
    
    features = [
        'svd_score', 'content_score', 'popularity_score', 
        'genre_overlap', 'context_genre_match', 'vote_confidence', 
        'user_maturity'
    ]
    
    print("--- SUMMARY STATISTICS ---")
    stats = df[features].describe().T[['mean', 'std', 'min', '50%', 'max']]
    print(stats)
    
    print("\n--- DATA QUALITY ---")
    missing = df[features].isnull().sum()
    print(f"Missing values:\n{missing[missing > 0] if not missing[missing > 0].empty else 'None'}")
    
    # Check constants
    unique_counts = df[features].nunique()
    constants = unique_counts[unique_counts <= 1].index.tolist()
    print(f"Constant features: {constants if constants else 'None'}")
    
    print("\n--- CORRELATION WITH LABEL ---")
    correlations = df[features + ['label']].corr()['label'].sort_values(ascending=False)
    print(correlations)
    
    print("\n--- POSITIVE VS NEGATIVE DISTRIBUTIONS (Mean) ---")
    pos_mean = df[df['label'] == 1][features].mean()
    neg_mean = df[df['label'] == 0][features].mean()
    # Avoid div by zero
    diff = (pos_mean - neg_mean) / (neg_mean + 1e-9) * 100
    
    comp_df = pd.DataFrame({
        'Pos Mean': pos_mean,
        'Neg Mean': neg_mean,
        '% Diff': diff
    })
    print(comp_df.sort_values('% Diff', ascending=False))

if __name__ == "__main__":
    import os
    audit()
