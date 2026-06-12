from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings
from recommender.svd_model import svd_model
from recommender.faiss_index import faiss_index
from recommender.hybrid_model import hybrid_model

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Pre-loading models and metadata...")
    try:
        svd_model.load()
        faiss_index.load()
        hybrid_model.load_metadata()
        print("Backend ready.")
    except Exception as e:
        print(f"Startup error: {e}")

app.include_router(api_router, prefix="/api")

@app.get("/")
async def health():
    return {"status": "healthy", "version": settings.VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
