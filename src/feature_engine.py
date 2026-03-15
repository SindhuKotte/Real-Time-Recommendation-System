"""
Feature engineering pipeline for the recommendation system.

Generates user-level, item-level, and interaction-level features
from raw Retail Rocket behavioral data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.utils import setup_logger

logger = setup_logger(__name__)


class FeatureEngine:
    """Build features for users, items, and user-item pairs."""

    def __init__(self, events: pd.DataFrame, interactions: pd.DataFrame,
                 item_features: Optional[pd.DataFrame] = None):
        self.events = events
        self.interactions = interactions
        self.item_features = item_features

    def build_user_features(self) -> pd.DataFrame:
        """Aggregate user-level behavioral features."""
        df = self.events.copy()

        user_agg = df.groupby("visitorid").agg(
            total_events=("event", "count"),
            n_unique_items=("itemid", "nunique"),
            n_views=("event", lambda x: (x == "view").sum()),
            n_addtocart=("event", lambda x: (x == "addtocart").sum()),
            n_transactions=("event", lambda x: (x == "transaction").sum()),
            first_event=("datetime", "min"),
            last_event=("datetime", "max"),
            n_sessions=("date", "nunique"),
            avg_hour=("hour", "mean"),
        ).reset_index()

        user_agg["view_to_cart_ratio"] = (
            user_agg["n_addtocart"] / user_agg["n_views"].clip(lower=1)
        )
        user_agg["cart_to_purchase_ratio"] = (
            user_agg["n_transactions"] / user_agg["n_addtocart"].clip(lower=1)
        )
        user_agg["view_to_purchase_ratio"] = (
            user_agg["n_transactions"] / user_agg["n_views"].clip(lower=1)
        )
        user_agg["activity_span_days"] = (
            (user_agg["last_event"] - user_agg["first_event"]).dt.total_seconds() / 86400
        )
        user_agg["events_per_session"] = (
            user_agg["total_events"] / user_agg["n_sessions"].clip(lower=1)
        )

        logger.info(f"User features: {user_agg.shape}")
        return user_agg

    def build_item_popularity_features(self) -> pd.DataFrame:
        """Aggregate item-level popularity and engagement features."""
        df = self.events.copy()

        item_agg = df.groupby("itemid").agg(
            total_events=("event", "count"),
            n_unique_users=("visitorid", "nunique"),
            n_views=("event", lambda x: (x == "view").sum()),
            n_addtocart=("event", lambda x: (x == "addtocart").sum()),
            n_transactions=("event", lambda x: (x == "transaction").sum()),
            first_seen=("datetime", "min"),
            last_seen=("datetime", "max"),
        ).reset_index()

        item_agg["conversion_rate"] = (
            item_agg["n_transactions"] / item_agg["n_views"].clip(lower=1)
        )
        item_agg["cart_rate"] = (
            item_agg["n_addtocart"] / item_agg["n_views"].clip(lower=1)
        )

        total_users = df["visitorid"].nunique()
        item_agg["user_penetration"] = item_agg["n_unique_users"] / total_users

        if self.item_features is not None:
            item_agg = item_agg.merge(self.item_features, on="itemid", how="left")

        logger.info(f"Item features: {item_agg.shape}")
        return item_agg

    def build_user_item_pair_features(self, user_features: pd.DataFrame,
                                       item_features: pd.DataFrame) -> pd.DataFrame:
        """Enrich interactions with user and item features for model input."""
        df = self.interactions.copy()

        df = df.merge(
            user_features.add_prefix("user_").rename(
                columns={"user_visitorid": "visitorid"}
            ),
            on="visitorid",
            how="left",
        )

        df = df.merge(
            item_features.add_prefix("item_").rename(
                columns={"item_itemid": "itemid"}
            ),
            on="itemid",
            how="left",
        )

        logger.info(f"Pair features: {df.shape}")
        return df

    def compute_temporal_features(self) -> pd.DataFrame:
        """Build time-based interaction features for session-aware recommendations."""
        df = self.events.sort_values(["visitorid", "timestamp"]).copy()

        df["time_since_last_event"] = df.groupby("visitorid")["timestamp"].diff()

        SESSION_GAP_MS = 30 * 60 * 1000  # 30 minutes
        df["new_session"] = (df["time_since_last_event"] > SESSION_GAP_MS).astype(int)
        df["session_id"] = df.groupby("visitorid")["new_session"].cumsum()

        session_features = df.groupby(["visitorid", "session_id"]).agg(
            session_length=("event", "count"),
            session_unique_items=("itemid", "nunique"),
            session_has_cart=("event", lambda x: int((x == "addtocart").any())),
            session_has_purchase=("event", lambda x: int((x == "transaction").any())),
            session_duration_ms=("timestamp", lambda x: x.max() - x.min()),
        ).reset_index()

        logger.info(f"Session features: {session_features.shape}")
        return session_features

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full feature engineering pipeline.

        Returns:
            Tuple of (user_features, item_features, enriched_interactions)
        """
        user_features = self.build_user_features()
        item_features = self.build_item_popularity_features()
        enriched = self.build_user_item_pair_features(user_features, item_features)
        return user_features, item_features, enriched
