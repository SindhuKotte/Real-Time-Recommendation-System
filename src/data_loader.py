"""
Data loading and preprocessing pipeline for Retail Rocket dataset.

The Retail Rocket dataset contains:
- events.csv: User behavioral events (view, addtocart, transaction)
- item_properties_part1.csv & part2.csv: Item property snapshots over time
- category_tree.csv: Hierarchical category structure
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional
from src.utils import load_config, setup_logger, ensure_dir, create_user_item_mapping

logger = setup_logger(__name__)


class RetailRocketDataLoader:
    """Load and preprocess the Retail Rocket e-commerce dataset."""

    EVENT_TYPES = ["view", "addtocart", "transaction"]

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Resolve project root: walk up from config file location
        config_path = self._find_config(config_path)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
        self.config = load_config(config_path)
        self.data_cfg = self.config["data"]
        self.preprocess_cfg = self.config["preprocessing"]

    @staticmethod
    def _find_config(config_path: str) -> str:
        """Search for config file relative to CWD or common parent directories."""
        if os.path.exists(config_path):
            return config_path
        # Try from parent directory (when running from notebooks/)
        parent = os.path.join("..", config_path)
        if os.path.exists(parent):
            return parent
        raise FileNotFoundError(f"Config not found at {config_path} or {parent}")

        self.events: Optional[pd.DataFrame] = None
        self.item_properties: Optional[pd.DataFrame] = None
        self.category_tree: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None

        self.user_to_idx: Dict = {}
        self.idx_to_user: Dict = {}
        self.item_to_idx: Dict = {}
        self.idx_to_item: Dict = {}

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve a config-relative path against the project root."""
        path = os.path.join(self.project_root, relative_path)
        if os.path.exists(path):
            return path
        return relative_path

    def load_raw_data(self) -> "RetailRocketDataLoader":
        """Load raw CSV files from the data directory."""
        raw_dir = self._resolve_path(self.data_cfg["raw_dir"])

        logger.info("Loading events data...")
        self.events = pd.read_csv(
            os.path.join(raw_dir, self.data_cfg["events_file"])
        )
        logger.info(f"  Events shape: {self.events.shape}")

        items_path_1 = os.path.join(raw_dir, self.data_cfg["items_file"])
        items_path_2 = os.path.join(raw_dir, self.data_cfg["items_file_2"])

        if os.path.exists(items_path_1):
            logger.info("Loading item properties...")
            parts = [pd.read_csv(items_path_1)]
            if os.path.exists(items_path_2):
                parts.append(pd.read_csv(items_path_2))
            self.item_properties = pd.concat(parts, ignore_index=True)
            logger.info(f"  Item properties shape: {self.item_properties.shape}")

        cat_path = os.path.join(raw_dir, self.data_cfg["category_tree_file"])
        if os.path.exists(cat_path):
            logger.info("Loading category tree...")
            self.category_tree = pd.read_csv(cat_path)
            logger.info(f"  Category tree shape: {self.category_tree.shape}")

        return self

    def preprocess_events(self) -> "RetailRocketDataLoader":
        """Clean and preprocess the events data."""
        if self.events is None:
            raise ValueError("Load raw data first with load_raw_data()")

        df = self.events.copy()
        logger.info(f"Raw events: {len(df):,}")

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek

        df = df.dropna(subset=["visitorid", "itemid", "event"])
        df["visitorid"] = df["visitorid"].astype(int)
        df["itemid"] = df["itemid"].astype(int)

        logger.info(f"After cleaning: {len(df):,}")
        logger.info(f"Unique users: {df['visitorid'].nunique():,}")
        logger.info(f"Unique items: {df['itemid'].nunique():,}")
        logger.info(f"Event distribution:\n{df['event'].value_counts()}")

        self.events = df
        return self

    def filter_sparse_users_items(self) -> "RetailRocketDataLoader":
        """Remove users and items with too few interactions for meaningful recommendations."""
        df = self.events.copy()
        min_user = self.preprocess_cfg["min_user_interactions"]
        min_item = self.preprocess_cfg["min_item_interactions"]

        initial_len = len(df)

        for iteration in range(5):
            user_counts = df["visitorid"].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            df = df[df["visitorid"].isin(valid_users)]

            item_counts = df["itemid"].value_counts()
            valid_items = item_counts[item_counts >= min_item].index
            df = df[df["itemid"].isin(valid_items)]

            if len(df) == len(self.events):
                break

        logger.info(
            f"Filtered {initial_len - len(df):,} sparse interactions "
            f"({len(df):,} remaining)"
        )
        logger.info(f"Users: {df['visitorid'].nunique():,}, Items: {df['itemid'].nunique():,}")

        self.events = df
        return self

    def create_implicit_ratings(self) -> "RetailRocketDataLoader":
        """Convert behavioral events to implicit rating scores.

        Weighting scheme:
        - view: 1.0
        - addtocart: 3.0
        - transaction: 5.0
        """
        weights = self.preprocess_cfg["interaction_weights"]

        df = self.events.copy()
        df["weight"] = df["event"].map(weights)

        interactions = (
            df.groupby(["visitorid", "itemid"])
            .agg(
                total_weight=("weight", "sum"),
                n_interactions=("event", "count"),
                last_event=("event", "last"),
                last_timestamp=("timestamp", "max"),
                events=("event", lambda x: list(x)),
            )
            .reset_index()
        )

        max_weight = interactions["total_weight"].quantile(0.99)
        interactions["rating"] = np.clip(
            interactions["total_weight"] / max_weight * 5, 0.5, 5.0
        )

        self.interactions = interactions
        logger.info(f"Created {len(interactions):,} user-item interactions")
        logger.info(f"Rating distribution:\n{interactions['rating'].describe()}")

        return self

    def build_mappings(self) -> "RetailRocketDataLoader":
        """Build contiguous user/item ID mappings."""
        if self.interactions is None:
            raise ValueError("Create implicit ratings first")

        self.user_to_idx, self.idx_to_user, self.item_to_idx, self.idx_to_item = (
            create_user_item_mapping(self.interactions)
        )

        self.interactions["user_idx"] = self.interactions["visitorid"].map(self.user_to_idx)
        self.interactions["item_idx"] = self.interactions["itemid"].map(self.item_to_idx)

        logger.info(
            f"Mappings built: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items"
        )
        return self

    def build_interaction_matrix(self) -> csr_matrix:
        """Build sparse user-item interaction matrix."""
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        matrix = csr_matrix(
            (
                self.interactions["rating"].values,
                (
                    self.interactions["user_idx"].values,
                    self.interactions["item_idx"].values,
                ),
            ),
            shape=(n_users, n_items),
        )

        sparsity = 1.0 - (matrix.nnz / (n_users * n_items))
        logger.info(f"Interaction matrix: {matrix.shape}, sparsity: {sparsity:.4%}")

        return matrix

    def build_item_features(self) -> Optional[pd.DataFrame]:
        """Extract item features from item properties for content-based filtering."""
        if self.item_properties is None:
            logger.warning("No item properties loaded")
            return None

        props = self.item_properties.copy()

        category_props = props[props["property"] == "categoryid"][
            ["itemid", "value"]
        ].drop_duplicates(subset=["itemid"], keep="last")
        category_props.columns = ["itemid", "categoryid"]

        available_props = props[props["property"] == "available"][
            ["itemid", "value"]
        ].drop_duplicates(subset=["itemid"], keep="last")
        available_props.columns = ["itemid", "available"]
        available_props["available"] = (
            available_props["available"].astype(str).str.strip().eq("1").astype(int)
        )

        item_features = category_props.merge(available_props, on="itemid", how="left")
        item_features["available"] = item_features["available"].fillna(0).astype(int)

        if self.category_tree is not None:
            item_features["categoryid"] = pd.to_numeric(
                item_features["categoryid"], errors="coerce"
            )
            self.category_tree["categoryid"] = pd.to_numeric(
                self.category_tree["categoryid"], errors="coerce"
            )
            item_features = item_features.merge(
                self.category_tree, on="categoryid", how="left"
            )

        logger.info(f"Item features built: {item_features.shape}")
        return item_features

    def temporal_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions temporally into train/val/test."""
        df = self.interactions.sort_values("last_timestamp")

        test_size = self.preprocess_cfg["test_size"]
        val_size = self.preprocess_cfg["val_size"]

        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()

        logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
        return train, val, test

    def save_processed(self) -> None:
        """Save processed data to disk."""
        out_dir = ensure_dir(self._resolve_path(self.data_cfg["processed_dir"]))

        if self.interactions is not None:
            self.interactions.to_parquet(
                os.path.join(out_dir, "interactions.parquet"), index=False
            )

        train, val, test = self.temporal_train_test_split()
        train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
        val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
        test.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

        item_features = self.build_item_features()
        if item_features is not None:
            item_features.to_parquet(
                os.path.join(out_dir, "item_features.parquet"), index=False
            )

        np.savez(
            os.path.join(out_dir, "mappings.npz"),
            user_to_idx=self.user_to_idx,
            idx_to_user=self.idx_to_user,
            item_to_idx=self.item_to_idx,
            idx_to_item=self.idx_to_item,
        )

        logger.info(f"Processed data saved to {out_dir}")

    def run_pipeline(self) -> "RetailRocketDataLoader":
        """Execute the full data loading and preprocessing pipeline."""
        return (
            self.load_raw_data()
            .preprocess_events()
            .filter_sparse_users_items()
            .create_implicit_ratings()
            .build_mappings()
        )
