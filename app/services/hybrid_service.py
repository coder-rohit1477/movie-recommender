from recommender.hybrid_model import hybrid_model

class HybridService:
    @staticmethod
    def get_recommendations(user_id: str, movie_title: str = None, top_k: int = 10):
        return hybrid_model.recommend(user_id, movie_title, top_k)
