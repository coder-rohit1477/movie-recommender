import os
import pickle
import logging
import pandas as pd
from app.core.config import settings

logger = logging.getLogger(__name__)

class LambdaRanker:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or os.path.join(settings.MODELS_DIR, "lambdarank_model.pkl")
        self.feature_cols = [
            'svd_score', 'content_score', 'popularity_score', 
            'genre_overlap', 'context_genre_match', 'user_maturity'
        ]

    def load(self):
        """Loads the serialized LightGBM model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"LambdaRank model not found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("LambdaRank model integrated successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load LambdaRank model: {e}")
            return False

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Generates ranking scores for a candidate matrix."""
        if self.model is None:
            raise RuntimeError("Ranker model not loaded.")
            
        # Ensure only training features are passed to LGBM
        X = features_df[self.feature_cols]
        return self.model.predict(X)

ranker = LambdaRanker()
