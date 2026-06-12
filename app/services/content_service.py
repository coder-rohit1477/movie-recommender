from recommender.faiss_index import faiss_index

class ContentService:
    @staticmethod
    def get_recommendations(movie_title: str, top_k: int = 10):
        return faiss_index.search(movie_title, top_k)
