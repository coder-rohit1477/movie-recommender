import sys
import os
import time
import pandas as pd
import random

# Ensure project root is in path
sys.path.append(os.getcwd())

from recommender.data_loader import DataLoader
from recommender.evaluation import create_temporal_split, Evaluator
from recommender.svd_model import SVDModel

def smoke_test():
    print("=== PART 1: SVD TRAINING VERIFICATION ===")
    try:
        start_time = time.time()
        print("Loading ratings data...")
        ratings = DataLoader.load_ratings()
        
        print("Creating temporal split...")
        train_df, test_df = create_temporal_split(ratings, test_ratio=0.2)
        
        print("Initializing model...")
        model = SVDModel()
        
        print("Fitting model (Dry Run)...")
        fit_start = time.time()
        model.fit(train_df)
        fit_end = time.time()
        
        print(f"Training SUCCESSful.")
        print(f"Training Time: {fit_end - fit_start:.2f} seconds.")
        part1_status = "PASS"
    except Exception as e:
        print(f"Training FAILED: {e}")
        part1_status = "FAIL"
        import traceback
        traceback.print_exc()
        return # Cannot continue if training fails

    print("\n=== PART 2: RECOMMENDATION VERIFICATION ===")
    try:
        sample_users = random.sample(list(train_df['userId'].unique()), 5)
        all_passed = True
        
        for user_id in sample_users:
            print(f"\nUser ID: {user_id}")
            recs = model.predict(user_id, top_n=10)
            print(f"Top 10 Recommendations: {list(recs.keys())}")
            
            # Check 1: Not empty
            if not recs:
                print("FAILED: Recommendations are empty.")
                all_passed = False
            
            # Check 2: Contains titles (not IDs or Nones)
            if any(not isinstance(t, str) or t == "Unknown" for t in recs.keys()):
                print("FAILED: Recommendations contain invalid titles.")
                all_passed = False
                
            # Check 3: Exclude already-rated movies
            user_rated_train = set(train_df[train_df['userId'] == user_id]['movieId'])
            # We need to map rec titles back to IDs or check if titles exist in train
            # Since model.predict checks against inner IDs, we verify logically:
            # Get titles of rated movies in train
            rated_titles = {model.movie_titles.get(mid) for mid in user_rated_train}
            intersection = set(recs.keys()) & rated_titles
            if intersection:
                print(f"FAILED: Recommendations include already-rated movies: {intersection}")
                all_passed = False
            else:
                print("Exclusion Check: PASS")

        part2_status = "PASS" if all_passed else "FAIL"
    except Exception as e:
        print(f"Recommendation Verification FAILED: {e}")
        part2_status = "FAIL"

    print("\n=== PART 3: EVALUATOR VERIFICATION ===")
    try:
        # Subset of 100 users for speed
        relevant_items_all = test_df[test_df['rating'] >= 3.5]['userId'].unique()
        subset_users = random.sample(list(relevant_items_all), min(100, len(relevant_items_all)))
        test_subset = test_df[test_df['userId'].isin(subset_users)]
        
        evaluator = Evaluator(k=10)
        print(f"Running Evaluator on {len(subset_users)} users...")
        eval_start = time.time()
        results = evaluator.run(model, train_df, test_subset)
        eval_end = time.time()
        
        print(f"Execution SUCCESSful. Time: {eval_end - eval_start:.2f} seconds.")
        part3_status = "PASS"
    except Exception as e:
        print(f"Evaluator FAILED: {e}")
        part3_status = "FAIL"
        import traceback
        traceback.print_exc()

    print("\n=== PART 4: RESULTS OBJECT ===")
    if part3_status == "PASS":
        print(f"Return Value from Evaluator.run():\n{results}")
        evaluator.report()
        part4_status = "PASS"
    else:
        part4_status = "FAIL"

    print("\n=== SMOKE TEST SUMMARY ===")
    print(f"PART 1 (Training):    {part1_status}")
    print(f"PART 2 (Recs):        {part2_status}")
    print(f"PART 3 (Evaluator):   {part3_status}")
    print(f"PART 4 (Results):     {part4_status}")

if __name__ == "__main__":
    smoke_test()
