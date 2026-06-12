from app.data_loader import load_ratings, load_movies

class PopularityModel:
    def __init__(self):
        self.popular_movies = []

    def compute(self):
        print("Computing Popularity (IMDb Weighted)...")
        ratings = load_ratings()
        movies = load_movies()
        
        if ratings.empty: return

        # Stats
        stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
        stats.columns = ['vote_count', 'vote_average']
        
        # Filter
        C = stats['vote_average'].mean()
        m = stats['vote_count'].quantile(0.90) # Top 10% most voted
        
        qualified = stats[stats['vote_count'] >= m].copy()
        
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v+m) * R) + (m / (v+m) * C)
        
        qualified['score'] = qualified.apply(weighted_rating, axis=1)
        
        # Merge Titles
        self.popular_movies = qualified.reset_index().merge(movies, on='movieId')
        self.popular_movies = self.popular_movies.sort_values('score', ascending=False)
        print(f"Popularity Computed: {len(self.popular_movies)} movies qualified.")

    def get_popular(self, top_k=10):
        if len(self.popular_movies) == 0:
            self.compute()
        
        if len(self.popular_movies) == 0: return []
        
        # Return simple list of dicts
        top = self.popular_movies.head(top_k)
        results = []
        for _, row in top.iterrows():
            results.append({
                "movieId": int(row['movieId']),
                "title": str(row['title']),
                "score": float(row['score']),
                "type": "popular"
            })
        return results

popularity_model = PopularityModel()
