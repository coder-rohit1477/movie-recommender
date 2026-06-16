import sys
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split
from recommender.hybrid_model import hybrid_model
from recommender.svd_model import svd_model
from recommender.faiss_index import faiss_index
from app.data_loader import load_movies

def generate():
    print("Loading data...")
    ratings = DataLoader.load_ratings()
    movies = load_movies()
    
    # Ensure movies has title mapping
    movie_title_map = dict(zip(movies['movieId'].astype(str), movies['title']))
    
    # Add title to ratings for easier labeling
    ratings['title'] = ratings['movieId'].astype(str).map(movie_title_map)
    
    print("Creating temporal split...")
    train_df, test_df = create_temporal_split(ratings)
    
    print("Fitting models on train split...")
    svd_model.fit(train_df, movies)
    faiss_index.load()
    
    # Inject train data into hybrid model for feature extraction
    hybrid_model.ratings_df = train_df
    hybrid_model.movies_df = movies
    hybrid_model.compute_popularity_scores()
    
    # Define Positive Pool from test set
    positives_map = test_df[test_df['rating'] >= 3.5].groupby('userId')['title'].apply(set).to_dict()
    
    # Select Users for Dataset Generation
    sampled_users = list(positives_map.keys())
    np.random.seed(42)
    sampled_users = np.random.choice(sampled_users, size=min(1000, len(sampled_users)), replace=False)
    
    dataset_rows = []
    group_sizes = []
    
    print(f"Generating features for {len(sampled_users)} users...")
    for user_id in tqdm(sampled_users):
        # Get context movie (latest from train)
        user_train = train_df[train_df['userId'] == user_id]
        if user_train.empty: continue
        
        latest_mid = user_train.sort_values('timestamp').iloc[-1]['movieId']
        context_title = movie_title_map.get(str(latest_mid))
        
        # 1. Retrieval Stage
        candidates_data = hybrid_model._retrieve_candidates(user_id, context_title)
        
        # 2. Feature Extraction Stage
        features_df = hybrid_model._get_candidate_features(
            user_id, 
            candidates_data["titles"], 
            candidates_data["cf_scores"], 
            candidates_data["cb_scores"], 
            context_title
        )
        
        if features_df.empty: continue
        
        # 3. Labeling
        user_positives = positives_map.get(user_id, set())
        features_df['label'] = features_df['title'].apply(lambda x: 1 if x in user_positives else 0)
        
        dataset_rows.append(features_df)
        group_sizes.append(len(features_df))

    # Combine and Save
    if not dataset_rows:
        print("Error: No data generated.")
        return
        
    full_df = pd.concat(dataset_rows)
    
    os.makedirs('data', exist_ok=True)
    full_df.to_csv('data/ranking_dataset.csv', index=False)
    
    with open('data/group_sizes.json', 'w') as f:
        json.dump(group_sizes, f)
        
    # Stats
    n_pos = full_df['label'].sum()
    n_total = len(full_df)
    print("\n--- DATASET STATISTICS ---")
    print(f"Users processed:    {len(sampled_users)}")
    print(f"Total candidates:   {n_total}")
    print(f"Total positives:    {n_pos}")
    print(f"Total negatives:    {n_total - n_pos}")
    print(f"Positive ratio:     {(n_pos/n_total)*100:.2f}%")
    
    print("\n--- SCHEMA ---")
    print(full_df.columns.tolist())
    print("\nExample Row (Positive):")
    print(full_df[full_df['label'] == 1].head(1).to_string())

if __name__ == "__main__":
    generate()
