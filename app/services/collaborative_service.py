from recommender.svd_model import svd_model

class CollaborativeService:
    @staticmethod
    def get_recommendations(user_id: str, top_k: int = 10):
        return svd_model.predict(user_id, top_n=top_k)

class ContentService:
    @staticmethod
    def get_recommendations(movie_title: str, top_k: int = 10):
        return faiss_index.search(movie_title, top_n=top_k)

# Need to import faiss_index in ContentService
from recommender.faiss_index import faiss_index
