"""Utility functions for the recommendation system."""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_sparsity(interaction_matrix: np.ndarray) -> float:
    """Calculate sparsity of an interaction matrix."""
    total = interaction_matrix.shape[0] * interaction_matrix.shape[1]
    nonzero = np.count_nonzero(interaction_matrix)
    return 1.0 - (nonzero / total)


def create_user_item_mapping(df: pd.DataFrame,
                              user_col: str = "visitorid",
                              item_col: str = "itemid") -> tuple:
    """Create contiguous ID mappings for users and items.

    Returns:
        Tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    """
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def train_test_split_temporal(df: pd.DataFrame,
                               timestamp_col: str = "timestamp",
                               test_size: float = 0.2) -> tuple:
    """Split data temporally — earlier interactions for training, later for testing."""
    df_sorted = df.sort_values(timestamp_col)
    split_idx = int(len(df_sorted) * (1 - test_size))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()
