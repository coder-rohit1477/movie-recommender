import numpy as np
from collections import defaultdict

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
