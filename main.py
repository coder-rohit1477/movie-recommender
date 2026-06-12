from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import sys
import os

# Add root to python path to ensure package resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender.hybrid_model import hybrid_model
from recommender.popularity_model import popularity_model
from recommender.data_loader import get_movie_titles
from recommender.cache import cache_response

app = FastAPI(title="Netflix-Grade Recommender API")

# CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "message": "Recommender System is Ready"}

@app.get("/recommend/hybrid")
@cache_response(ttl_seconds=300)
async def recommend_hybrid(
    user_id: int = Query(..., description="User ID"),
    movie_title: Optional[str] = Query(None, description="Context movie"),
    top_k: int = 10
):
    try:
        recs = hybrid_model.recommend(user_id=user_id, movie_title=movie_title, top_k=top_k)
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/trending")
@cache_response(ttl_seconds=3600)
async def recommend_trending(top_k: int = 10):
    # For now, trending == popular (could serve recent movies if release_date was parsed)
    return {"recommendations": popularity_model.get_popular(top_k)}

@app.get("/recommend/popular")
@cache_response(ttl_seconds=3600)
async def recommend_popular(top_k: int = 10):
    return {"recommendations": popularity_model.get_popular(top_k)}

@app.get("/recommend/explain")
async def explain():
    return {"message": "Explanation: Hybrid model combines SVD (User History) + FAISS (Content Similarity) + IMDb Weighted Rating (Popularity)."}

@app.get("/movies/search")
async def search_movies(query: str, limit: int = 10):
    titles = get_movie_titles()
    # Simple substring search
    results = []
    q = query.lower()
    for mid, title in titles.items():
        if q in str(title).lower():
            results.append({"movie_id": mid, "title": title})
            if len(results) >= limit: break
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    # Pre-load models on startup
    print("Pre-loading models...")
    popularity_model.compute()
    uvicorn.run(app, host="0.0.0.0", port=8000)
