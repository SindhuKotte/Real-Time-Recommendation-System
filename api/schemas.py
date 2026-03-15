"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RecommendationStrategy(str, Enum):
    HYBRID = "hybrid"
    COLLABORATIVE = "collaborative"
    MATRIX_FACTORIZATION = "matrix_factorization"
    CONTENT_BASED = "content_based"
    NCF = "ncf"
    POPULAR = "popular"


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to generate recommendations for")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    strategy: RecommendationStrategy = Field(
        default=RecommendationStrategy.HYBRID,
        description="Recommendation strategy to use"
    )

    model_config = {"json_schema_extra": {
        "examples": [{"user_id": 123456, "top_n": 10, "strategy": "hybrid"}]
    }}


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str
    recommendations: List[RecommendationItem]
    n_results: int


class SimilarItemsRequest(BaseModel):
    item_id: int = Field(..., description="Item ID to find similar items for")
    top_n: int = Field(default=10, ge=1, le=100)


class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[RecommendationItem]
    n_results: int


class UserHistoryItem(BaseModel):
    item_id: int
    rating: float
    n_interactions: int


class UserHistoryResponse(BaseModel):
    user_id: int
    history: List[UserHistoryItem]
    n_items: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    n_users: int
    n_items: int
