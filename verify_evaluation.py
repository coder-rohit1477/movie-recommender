import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split

def verify():
    print("Loading data...")
    ratings = DataLoader.load_ratings()
    
    initial_users = ratings['userId'].nunique()
    initial_ratings = len(ratings)
    print(f"Initial Ratings: {initial_ratings}")
    print(f"Initial Users: {initial_users}")

    print("\nExecuting Temporal Split (80/20)...")
    train_df, test_df = create_temporal_split(ratings, test_ratio=0.2)
    
    train_users = set(train_df['userId'].unique())
    test_users = set(test_df['userId'].unique())
    
    # Analysis
    skipped_users = initial_users - len(train_users.union(test_users))
    test_users_missing_in_train = test_users - train_users
    
    print("\n--- RESULTS ---")
    print(f"1. Train Set Size: {len(train_df)} ratings")
    print(f"   Test Set Size:  {len(test_df)} ratings")
    print(f"2. Train Users:    {len(train_users)}")
    print(f"3. Test Users:     {len(test_users)}")
    print(f"4. Skipped Users:  {skipped_users} (history < 5 or split logic)")
    print(f"5. Test users NOT in train: {len(test_users_missing_in_train)}")
    
    # Temporal Check
    print("\nVerifying Timestamps (Vectorized)...")
    # Join train max and test min to verify order
    train_max = train_df.groupby('userId')['timestamp'].max().rename('train_max')
    test_min = test_df.groupby('userId')['timestamp'].min().rename('test_min')
    
    check_df = pd.concat([train_max, test_min], axis=1).dropna()
    violations = check_df[check_df['train_max'] > check_df['test_min']]
    
    if len(violations) == 0:
        print("6. Timestamp Order: VALID (Train <= Test for all users)")
    else:
        print(f"6. Timestamp Order: INVALID ({len(violations)} violations found)")

if __name__ == "__main__":
    verify()
