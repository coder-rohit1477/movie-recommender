import pandas as pd
import json
import os

def verify():
    if not os.path.exists('data/ranking_dataset.csv'):
        print("Error: data/ranking_dataset.csv not found.")
        return

    df = pd.read_csv('data/ranking_dataset.csv')
    with open('data/group_sizes.json', 'r') as f:
        groups = json.load(f)
    
    n_train_users = int(0.8 * len(groups))
    train_row_limit = sum(groups[:n_train_users])
    
    train_df = df.iloc[:train_row_limit]
    val_df = df.iloc[train_row_limit:]
    
    train_users = set(train_df['user_id'].unique())
    val_users = set(val_df['user_id'].unique())
    
    print(f"--- SPLIT VERIFICATION ---")
    print(f"Train Users: {len(train_users)}")
    print(f"Val Users:   {len(val_users)}")
    
    # Check 1: User Disjointness
    overlap = train_users.intersection(val_users)
    print(f"Overlap Users: {len(overlap)}")
    
    # Check 2: Row Count vs Groups
    print(f"Total Rows: {len(df)}")
    print(f"Sum of Groups: {sum(groups)}")
    
    # Check 3: Duplicates
    dupes = df.duplicated().sum()
    print(f"Duplicate rows in dataset: {dupes}")
    
    if len(overlap) == 0 and sum(groups) == len(df) and dupes == 0:
        print("\nInternal Verdict: PASS")
    else:
        print("\nInternal Verdict: FAIL")

if __name__ == "__main__":
    verify()
