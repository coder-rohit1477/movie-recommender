from fastapi import APIRouter, Query, HTTPException
from app.services.hybrid_service import HybridService
from app.services.collaborative_service import CollaborativeService
from app.services.content_service import ContentService
from app.core.cache import cache
from recommender.hybrid_model import hybrid_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/recommend/hybrid")
async def get_hybrid_recommendations(
    user_id: str = Query(..., description="The ID of the user"),
    movie_title: str = Query(None, description="Optional movie title to base content similarity on"),
    top_k: int = Query(10, ge=1, le=50)
):
    """
    Hybrid recommendation endpoint combining SVD (collaborative) and FAISS (content).
    Validates movie existence and handles unknown users gracefully.
    """
    # 1. Validate Movie Title if provided
    if movie_title and not hybrid_model.validate_movie(movie_title):
        logger.warning(f"Validation failed: Movie '{movie_title}' not found.")
        raise HTTPException(
            status_code=404, 
            detail=f"Movie '{movie_title}' not found in our database."
        )

    # 2. Check Cache
    cache_key = f"hybrid_{user_id}_{movie_title}_{top_k}"
    try:
        cached = cache.get(cache_key)
        if cached:
            return {"recommendations": cached, "source": "cache"}
    except Exception as e:
        logger.error(f"Cache error: {e}")
    
    # 3. Generate Recommendations
    try:
        recs = HybridService.get_recommendations(user_id, movie_title, top_k)
        
        # 4. Update Cache (Fire and forget or handle error)
        try:
            cache.set(cache_key, recs)
        except Exception as e:
            logger.error(f"Failed to update cache: {e}")
            
        return {"recommendations": recs, "source": "inference"}

    except ValueError as e:
        logger.error(f"Value error in hybrid recommendation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in hybrid endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get("/recommend/trending")
@router.get("/movies/trending")
async def get_trending_movies(top_k: int = Query(10, ge=1, le=50)):
    # Using pre-computed popularity scores
    sorted_pop = sorted(hybrid_model.popularity_scores.items(), key=lambda x: x[1], reverse=True)
    return {"movies": [title for title, _ in sorted_pop[:top_k]]}

@router.get("/recommend/popular")
@router.get("/recommend/collaborative")
async def get_collaborative_recommendations(
    user_id: str = Query(...),
    top_k: int = Query(10, ge=1, le=50)
):
    cache_key = f"collab_{user_id}_{top_k}"
    cached = cache.get(cache_key)
    if cached:
        return {"recommendations": cached, "source": "cache"}
    
    try:
        recs = CollaborativeService.get_recommendations(user_id, top_k)
        cache.set(cache_key, recs)
        return {"recommendations": recs, "source": "inference"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movies/search")
async def search_movies(q: str = Query(..., min_length=2)):
    titles = hybrid_model.movies_df['title'].tolist()
    matches = [t for t in titles if q.lower() in t.lower()]
    return {"results": matches[:10]}
