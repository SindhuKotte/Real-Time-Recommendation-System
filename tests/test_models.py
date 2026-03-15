"""Unit tests for recommendation models."""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.collaborative import CollaborativeFilteringKNN
from src.models.content_based import ContentBasedModel
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k


@pytest.fixture
def sample_interaction_matrix():
    """Create a small interaction matrix for testing."""
    data = np.array([
        [5, 3, 0, 1, 0],
        [4, 0, 0, 1, 0],
        [1, 1, 0, 5, 0],
        [0, 0, 5, 4, 0],
        [0, 1, 4, 0, 5],
    ], dtype=np.float64)
    return csr_matrix(data)


@pytest.fixture
def sample_interactions():
    """Create sample interaction DataFrame."""
    return pd.DataFrame({
        "visitorid": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
        "itemid": [100, 101, 102, 100, 103, 101, 102, 103, 104, 104],
        "rating": [5.0, 3.0, 1.0, 4.0, 1.0, 1.0, 5.0, 4.0, 5.0, 5.0],
        "user_idx": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3],
        "item_idx": [0, 1, 2, 0, 3, 1, 2, 3, 4, 4],
    })


class TestCollaborativeFiltering:
    def test_fit(self, sample_interaction_matrix):
        model = CollaborativeFilteringKNN(k=3, mode="item")
        model.fit(sample_interaction_matrix)
        assert model.model is not None
        assert model.n_users == 5
        assert model.n_items == 5

    def test_predict(self, sample_interaction_matrix):
        model = CollaborativeFilteringKNN(k=3, mode="item")
        model.fit(sample_interaction_matrix)
        recs = model.predict_for_user(0, top_n=3)
        assert isinstance(recs, list)
        for item_idx, score in recs:
            assert isinstance(item_idx, (int, np.integer))
            assert isinstance(score, (float, np.floating))

    def test_similar_items(self, sample_interaction_matrix):
        model = CollaborativeFilteringKNN(k=3, mode="item")
        model.fit(sample_interaction_matrix)
        similar = model.get_similar_items(0, top_n=3)
        assert len(similar) <= 3


class TestContentBased:
    def test_fit_without_features(self, sample_interactions):
        model = ContentBasedModel()
        model.fit(sample_interactions)
        assert model.item_similarity is not None

    def test_predict(self, sample_interactions):
        model = ContentBasedModel()
        model.fit(sample_interactions)
        recs = model.predict_for_user(1, top_n=3)
        assert isinstance(recs, list)


class TestEvaluationMetrics:
    def test_precision_at_k(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 6}
        assert precision_at_k(recommended, relevant, 5) == 2 / 5

    def test_recall_at_k(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 6}
        assert recall_at_k(recommended, relevant, 5) == 2 / 3

    def test_ndcg_at_k(self):
        recommended = [1, 2, 3]
        relevant = {1}
        score = ndcg_at_k(recommended, relevant, 3)
        assert score == 1.0  # best item at rank 1

    def test_hit_rate(self):
        assert hit_rate_at_k([1, 2, 3], {2}, 3) == 1.0
        assert hit_rate_at_k([1, 2, 3], {4}, 3) == 0.0

    def test_empty_recommendations(self):
        assert precision_at_k([], {1, 2}, 5) == 0.0
        assert recall_at_k([1], set(), 5) == 0.0
        assert ndcg_at_k([], {1}, 5) == 0.0
