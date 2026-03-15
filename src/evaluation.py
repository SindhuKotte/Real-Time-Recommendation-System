"""
Evaluation framework for recommendation models.

Implements standard information retrieval and recommendation metrics:
- Precision@K, Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- Hit Rate@K
- Coverage & Diversity
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict
from src.utils import setup_logger

logger = setup_logger(__name__)


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Fraction of recommended items in top-K that are relevant."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = len(set(rec_k) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Fraction of relevant items that appear in top-K recommendations."""
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Average Precision at K — rewards relevant items appearing earlier."""
    rec_k = recommended[:k]
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(len(relevant), k) if relevant else 0.0


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """Binary: 1 if any relevant item appears in top-K, else 0."""
    rec_k = set(recommended[:k])
    return 1.0 if rec_k & relevant else 0.0


def coverage(all_recommendations: List[List[int]], n_total_items: int) -> float:
    """Fraction of total catalog covered by all recommendations."""
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / n_total_items if n_total_items > 0 else 0.0


def diversity(all_recommendations: List[List[int]]) -> float:
    """Average pairwise dissimilarity across recommendation lists (1 - Jaccard)."""
    if len(all_recommendations) < 2:
        return 0.0

    total_dissimilarity = 0.0
    n_pairs = 0

    for i in range(len(all_recommendations)):
        for j in range(i + 1, len(all_recommendations)):
            set_i = set(all_recommendations[i])
            set_j = set(all_recommendations[j])
            union = set_i | set_j
            if union:
                jaccard = len(set_i & set_j) / len(union)
                total_dissimilarity += 1 - jaccard
                n_pairs += 1

    return total_dissimilarity / n_pairs if n_pairs > 0 else 0.0


def novelty(recommended: List[int], item_popularity: Dict[int, float],
            n_total_users: int) -> float:
    """Average self-information of recommended items — higher means more novel."""
    if not recommended or n_total_users == 0:
        return 0.0
    scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1) / n_total_users
        if pop > 0:
            scores.append(-np.log2(pop))
    return np.mean(scores) if scores else 0.0


class RecommendationEvaluator:
    """Comprehensive evaluator for recommendation models."""

    def __init__(self, top_k_values: List[int] = None):
        self.top_k_values = top_k_values or [5, 10, 20]

    def evaluate_model(self, model: Any, test_interactions: pd.DataFrame,
                        n_total_items: int, model_name: str = "model",
                        user_col: str = "visitorid", item_col: str = "itemid",
                        user_idx_col: Optional[str] = "user_idx",
                        use_idx: bool = False,
                        max_users: int = 500) -> Dict[str, float]:
        """Evaluate a model on test data across all metrics and K values.

        Args:
            model: Trained model with predict_for_user() method
            test_interactions: Test DataFrame with user-item pairs
            n_total_items: Total items in catalog
            model_name: For logging
            use_idx: If True, pass user_idx to model instead of user_id
            max_users: Max users to evaluate (samples randomly for speed)
        """
        ground_truth = self._build_ground_truth(test_interactions, user_col, item_col)

        results = {}
        all_recs_by_k = defaultdict(list)
        max_k = max(self.top_k_values)

        user_ids = list(ground_truth.keys())
        if len(user_ids) > max_users:
            np.random.seed(42)
            user_ids = list(np.random.choice(user_ids, max_users, replace=False))
        logger.info(f"Evaluating {model_name} on {len(user_ids)} users...")

        user_predictions = {}
        for uid in user_ids:
            try:
                if use_idx and user_idx_col:
                    user_rows = test_interactions[test_interactions[user_col] == uid]
                    if user_idx_col in user_rows.columns and len(user_rows) > 0:
                        uidx = int(user_rows[user_idx_col].iloc[0])
                        recs = model.predict_for_user(uidx, top_n=max_k)
                    else:
                        continue
                else:
                    recs = model.predict_for_user(uid, top_n=max_k)

                rec_items = [item_id for item_id, _ in recs]
                user_predictions[uid] = rec_items
            except Exception as e:
                logger.debug(f"Skipping user {uid}: {e}")
                continue

        for k in self.top_k_values:
            p_scores, r_scores, n_scores, ap_scores, hr_scores = [], [], [], [], []

            for uid, rec_items in user_predictions.items():
                relevant = ground_truth[uid]
                p_scores.append(precision_at_k(rec_items, relevant, k))
                r_scores.append(recall_at_k(rec_items, relevant, k))
                n_scores.append(ndcg_at_k(rec_items, relevant, k))
                ap_scores.append(average_precision_at_k(rec_items, relevant, k))
                hr_scores.append(hit_rate_at_k(rec_items, relevant, k))
                all_recs_by_k[k].append(rec_items[:k])

            results[f"precision@{k}"] = np.mean(p_scores) if p_scores else 0.0
            results[f"recall@{k}"] = np.mean(r_scores) if r_scores else 0.0
            results[f"ndcg@{k}"] = np.mean(n_scores) if n_scores else 0.0
            results[f"map@{k}"] = np.mean(ap_scores) if ap_scores else 0.0
            results[f"hit_rate@{k}"] = np.mean(hr_scores) if hr_scores else 0.0

        for k in self.top_k_values:
            results[f"coverage@{k}"] = coverage(all_recs_by_k[k], n_total_items)
            results[f"diversity@{k}"] = diversity(all_recs_by_k[k][:200])

        results["n_evaluated_users"] = len(user_predictions)
        logger.info(f"\n{model_name} Results:")
        for metric, value in sorted(results.items()):
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")

        return results

    @staticmethod
    def _build_ground_truth(test_df: pd.DataFrame, user_col: str,
                             item_col: str) -> Dict[int, Set[int]]:
        """Build ground truth mapping: user → set of relevant items."""
        gt = defaultdict(set)
        for _, row in test_df.iterrows():
            gt[row[user_col]].add(row[item_col])
        return dict(gt)

    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a comparison DataFrame from multiple model results."""
        df = pd.DataFrame(results).T
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        return df.sort_values(df.columns[0], ascending=False)
