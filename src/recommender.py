"""
Unified Recommender interface.

Provides a single entry point for loading trained models, generating
recommendations, and serving predictions via the API layer.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from src.utils import load_config, setup_logger
from src.models.collaborative import CollaborativeFilteringKNN
from src.models.matrix_factor import MatrixFactorizationModel
from src.models.content_based import ContentBasedModel
from src.models.ncf import NCFTrainer
from src.models.hybrid import HybridRecommender

logger = setup_logger(__name__)


class RecommendationEngine:
    """Unified interface for all recommendation models.

    Handles model loading, prediction routing, and fallback logic
    for the API and dashboard layers.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.models: Dict[str, Any] = {}
        self.hybrid: Optional[HybridRecommender] = None

        self.interactions: Optional[pd.DataFrame] = None
        self.user_to_idx: Dict = {}
        self.idx_to_user: Dict = {}
        self.item_to_idx: Dict = {}
        self.idx_to_item: Dict = {}
        self.item_popularity: Dict[int, int] = {}

    def load_data(self, data_dir: str = "data/processed") -> "RecommendationEngine":
        """Load processed interactions and mappings."""
        self.interactions = pd.read_parquet(os.path.join(data_dir, "interactions.parquet"))

        mappings = np.load(os.path.join(data_dir, "mappings.npz"), allow_pickle=True)
        self.user_to_idx = mappings["user_to_idx"].item()
        self.idx_to_user = mappings["idx_to_user"].item()
        self.item_to_idx = mappings["item_to_idx"].item()
        self.idx_to_item = mappings["idx_to_item"].item()

        self.item_popularity = (
            self.interactions.groupby("itemid")["rating"]
            .count()
            .to_dict()
        )

        logger.info(
            f"Loaded data: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items"
        )
        return self

    def load_models(self, model_dir: str = "models") -> "RecommendationEngine":
        """Load all trained models from disk."""
        cf_path = os.path.join(model_dir, "collaborative_filtering.pkl")
        if os.path.exists(cf_path):
            with open(cf_path, "rb") as f:
                self.models["collaborative"] = pickle.load(f)
            logger.info("Loaded collaborative filtering model")

        mf_path = os.path.join(model_dir, "matrix_factorization.pkl")
        if os.path.exists(mf_path):
            with open(mf_path, "rb") as f:
                self.models["matrix_factorization"] = pickle.load(f)
            logger.info("Loaded matrix factorization model")

        cb_path = os.path.join(model_dir, "content_based.pkl")
        if os.path.exists(cb_path):
            with open(cb_path, "rb") as f:
                self.models["content_based"] = pickle.load(f)
            logger.info("Loaded content-based model")

        ncf_path = os.path.join(model_dir, "ncf_model.pt")
        if os.path.exists(ncf_path):
            ncf_config = self.config["models"]["ncf"]
            n_users = len(self.user_to_idx)
            n_items = len(self.item_to_idx)
            ncf_trainer = NCFTrainer(n_users, n_items, ncf_config)
            ncf_trainer.load_model(ncf_path)
            self.models["ncf"] = ncf_trainer
            logger.info("Loaded NCF model")

        if self.models:
            self.hybrid = HybridRecommender(
                weights=self.config["models"]["hybrid"]["weights"]
            )
            self.hybrid.set_mappings(self.idx_to_item, self.item_to_idx)
            for name, model in self.models.items():
                self.hybrid.register_model(name, model)

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        return self

    def recommend(self, user_id: int, top_n: int = 10,
                   strategy: str = "hybrid") -> List[Dict]:
        """Generate recommendations for a user.

        Args:
            user_id: Original user ID
            top_n: Number of recommendations
            strategy: 'hybrid', 'collaborative', 'matrix_factorization',
                      'content_based', 'ncf', or 'popular'

        Returns:
            List of dicts with 'item_id', 'score', 'rank'
        """
        if strategy == "popular":
            return self._popular_recommendations(top_n, user_id)

        user_idx = self.user_to_idx.get(user_id)

        if strategy == "hybrid" and self.hybrid:
            recs = self.hybrid.predict_for_user(user_id, user_idx, top_n)
        elif strategy in self.models:
            model = self.models[strategy]
            if strategy in ("collaborative", "ncf") and user_idx is not None:
                recs = model.predict_for_user(user_idx, top_n=top_n)
                recs = [(self.idx_to_item.get(idx, idx), score) for idx, score in recs]
            elif strategy in ("matrix_factorization", "content_based"):
                recs = model.predict_for_user(user_id, top_n=top_n)
            else:
                return self._popular_recommendations(top_n, user_id)
        else:
            return self._popular_recommendations(top_n, user_id)

        if not recs:
            return self._popular_recommendations(top_n, user_id)

        return [
            {"item_id": int(item_id), "score": round(float(score), 4), "rank": rank + 1}
            for rank, (item_id, score) in enumerate(recs)
        ]

    def _popular_recommendations(self, top_n: int,
                                   exclude_user: Optional[int] = None) -> List[Dict]:
        """Fallback: recommend most popular items."""
        user_items = set()
        if exclude_user is not None and self.interactions is not None:
            user_items = set(
                self.interactions[self.interactions["visitorid"] == exclude_user]["itemid"]
            )

        sorted_items = sorted(
            self.item_popularity.items(), key=lambda x: x[1], reverse=True
        )

        recs = []
        for item_id, pop_score in sorted_items:
            if item_id not in user_items:
                recs.append({
                    "item_id": int(item_id),
                    "score": round(float(pop_score), 4),
                    "rank": len(recs) + 1,
                })
                if len(recs) >= top_n:
                    break

        return recs

    def get_similar_items(self, item_id: int, top_n: int = 10) -> List[Dict]:
        """Find items similar to a given item."""
        if "content_based" in self.models:
            recs = self.models["content_based"].get_similar_items(item_id, top_n)
            return [
                {"item_id": int(iid), "score": round(float(s), 4), "rank": r + 1}
                for r, (iid, s) in enumerate(recs)
            ]

        if "collaborative" in self.models:
            item_idx = self.item_to_idx.get(item_id)
            if item_idx is not None:
                recs = self.models["collaborative"].get_similar_items(item_idx, top_n)
                return [
                    {"item_id": int(self.idx_to_item.get(idx, idx)),
                     "score": round(float(s), 4), "rank": r + 1}
                    for r, (idx, s) in enumerate(recs)
                ]

        return []

    def get_user_history(self, user_id: int) -> List[Dict]:
        """Get a user's interaction history."""
        if self.interactions is None:
            return []

        history = self.interactions[self.interactions["visitorid"] == user_id]
        return [
            {
                "item_id": int(row["itemid"]),
                "rating": round(float(row["rating"]), 2),
                "n_interactions": int(row["n_interactions"]),
            }
            for _, row in history.iterrows()
        ]

    def get_available_strategies(self) -> List[str]:
        """List available recommendation strategies."""
        strategies = list(self.models.keys())
        if self.hybrid:
            strategies.append("hybrid")
        strategies.append("popular")
        return strategies
