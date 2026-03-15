"""
FastAPI real-time recommendation API.

Endpoints:
    GET  /health                     — Service health check
    POST /recommend                  — Get personalized recommendations
    POST /similar-items              — Get items similar to a given item
    GET  /user/{user_id}/history     — Get user interaction history
    GET  /strategies                 — List available recommendation strategies
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    RecommendationRequest, RecommendationResponse, RecommendationItem,
    SimilarItemsRequest, SimilarItemsResponse,
    UserHistoryResponse, UserHistoryItem,
    HealthResponse,
)
from src.recommender import RecommendationEngine
from src.utils import load_config, setup_logger

logger = setup_logger(__name__)
engine = RecommendationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Loading recommendation engine...")
    try:
        engine.load_data()
        engine.load_models()
        logger.info("Recommendation engine ready")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        logger.info("API running without models (health endpoint available)")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="E-Commerce Recommendation API",
    description=(
        "Real-time product recommendation engine supporting multiple strategies: "
        "Collaborative Filtering, Matrix Factorization (SVD), Content-Based, "
        "Neural Collaborative Filtering (PyTorch), and Hybrid Ensemble."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health and model status."""
    return HealthResponse(
        status="healthy",
        models_loaded=engine.get_available_strategies(),
        n_users=len(engine.user_to_idx),
        n_items=len(engine.item_to_idx),
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Generate personalized recommendations for a user."""
    start = time.time()

    if request.user_id not in engine.user_to_idx and request.strategy != "popular":
        if engine.interactions is not None:
            if request.user_id not in engine.interactions["visitorid"].values:
                raise HTTPException(
                    status_code=404,
                    detail=f"User {request.user_id} not found. Use strategy='popular' for new users."
                )

    recs = engine.recommend(
        user_id=request.user_id,
        top_n=request.top_n,
        strategy=request.strategy.value,
    )

    latency_ms = (time.time() - start) * 1000
    logger.info(
        f"Recommendations for user {request.user_id} "
        f"({request.strategy.value}): {len(recs)} items in {latency_ms:.1f}ms"
    )

    return RecommendationResponse(
        user_id=request.user_id,
        strategy=request.strategy.value,
        recommendations=[RecommendationItem(**r) for r in recs],
        n_results=len(recs),
    )


@app.post("/similar-items", response_model=SimilarItemsResponse)
async def get_similar_items(request: SimilarItemsRequest):
    """Find items similar to a given item."""
    results = engine.get_similar_items(request.item_id, request.top_n)

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Item {request.item_id} not found or no similar items available."
        )

    return SimilarItemsResponse(
        item_id=request.item_id,
        similar_items=[RecommendationItem(**r) for r in results],
        n_results=len(results),
    )


@app.get("/user/{user_id}/history", response_model=UserHistoryResponse)
async def get_user_history(user_id: int):
    """Get a user's interaction history."""
    history = engine.get_user_history(user_id)

    if not history:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    return UserHistoryResponse(
        user_id=user_id,
        history=[UserHistoryItem(**h) for h in history],
        n_items=len(history),
    )


@app.get("/strategies")
async def list_strategies():
    """List available recommendation strategies."""
    return {"strategies": engine.get_available_strategies()}


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(
        "api.app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True,
    )
