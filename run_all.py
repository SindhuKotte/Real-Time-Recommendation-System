"""Run the full pipeline: data loading, all 5 models, evaluation, MLflow tracking."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import mlflow
from scipy.sparse import csr_matrix
from collections import defaultdict

from src.data_loader import RetailRocketDataLoader
from src.feature_engine import FeatureEngine
from src.models.collaborative import CollaborativeFilteringKNN
from src.models.matrix_factor import MatrixFactorizationModel
from src.models.content_based import ContentBasedModel
from src.models.ncf import NCFTrainer
from src.models.hybrid import HybridRecommender
from src.evaluation import (
    RecommendationEvaluator, precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k
)
from src.utils import load_config, ensure_dir

config = load_config("configs/config.yaml")
MODEL_DIR = ensure_dir("models")

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

print("=" * 60)
print("STEP 1/6: Loading & Preprocessing Data")
print("=" * 60)

loader = RetailRocketDataLoader()
loader.run_pipeline()
train, val, test = loader.temporal_train_test_split()
item_features = loader.build_item_features()

n_users = len(loader.user_to_idx)
n_items = len(loader.item_to_idx)
print(f"Users: {n_users:,}, Items: {n_items:,}")
print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

train_matrix = csr_matrix(
    (train["rating"].values, (train["user_idx"].values, train["item_idx"].values)),
    shape=(n_users, n_items),
)

evaluator = RecommendationEvaluator(top_k_values=config["evaluation"]["top_k"])

# ── Model 1: Collaborative Filtering ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2/6: Collaborative Filtering (Item-KNN)")
print("=" * 60)

cf_cfg = config["models"]["collaborative_filtering"]
with mlflow.start_run(run_name="collaborative_filtering"):
    cf_model = CollaborativeFilteringKNN(
        k=cf_cfg["k_neighbors"], metric=cf_cfg["similarity"],
        min_k=cf_cfg["min_k"], mode="item",
    )
    cf_model.fit(train_matrix)
    mlflow.log_params(cf_model.get_params())
    cf_results = evaluator.evaluate_model(
        cf_model, test, n_items, "CF-KNN", user_idx_col="user_idx", use_idx=True,
    )
    for k, v in cf_results.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k.replace("@", "_at_"), v)
    with open(os.path.join(MODEL_DIR, "collaborative_filtering.pkl"), "wb") as f:
        pickle.dump(cf_model, f)
print("Done!")

# ── Model 2: Matrix Factorization (SVD) ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 3/6: Matrix Factorization (SVD)")
print("=" * 60)

mf_cfg = config["models"]["matrix_factorization"]
with mlflow.start_run(run_name="matrix_factorization_svd"):
    mf_model = MatrixFactorizationModel(
        algorithm=mf_cfg["algorithm"], n_factors=mf_cfg["n_factors"],
        n_epochs=mf_cfg["n_epochs"], lr_all=mf_cfg["lr_all"],
        reg_all=mf_cfg["reg_all"],
    )
    mf_model.fit(train)
    mlflow.log_params(mf_model.get_params())
    mf_results = evaluator.evaluate_model(
        mf_model, test, n_items, "MF-SVD", use_idx=False,
    )
    for k, v in mf_results.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k.replace("@", "_at_"), v)
    with open(os.path.join(MODEL_DIR, "matrix_factorization.pkl"), "wb") as f:
        pickle.dump(mf_model, f)
print("Done!")

# ── Model 3: Content-Based ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4/6: Content-Based Filtering")
print("=" * 60)

with mlflow.start_run(run_name="content_based"):
    cb_model = ContentBasedModel(
        similarity_metric=config["models"]["content_based"]["similarity_metric"],
    )
    cb_model.fit(train, item_features)
    mlflow.log_params(cb_model.get_params())
    cb_results = evaluator.evaluate_model(
        cb_model, test, n_items, "Content-Based", use_idx=False,
    )
    for k, v in cb_results.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k.replace("@", "_at_"), v)
    with open(os.path.join(MODEL_DIR, "content_based.pkl"), "wb") as f:
        pickle.dump(cb_model, f)
print("Done!")

# ── Model 4: Neural Collaborative Filtering (PyTorch) ────────────────
print("\n" + "=" * 60)
print("STEP 5/6: Neural Collaborative Filtering (PyTorch)")
print("=" * 60)

ncf_cfg = config["models"]["ncf"]
with mlflow.start_run(run_name="ncf_pytorch"):
    ncf_trainer = NCFTrainer(n_users, n_items, ncf_cfg)
    ncf_trainer.fit(train, val, epochs=ncf_cfg["epochs"])
    mlflow.log_params(ncf_trainer.get_params())
    for epoch, loss in enumerate(ncf_trainer.train_losses):
        mlflow.log_metric("train_loss", loss, step=epoch)
    for epoch, loss in enumerate(ncf_trainer.val_losses):
        mlflow.log_metric("val_loss", loss, step=epoch)
    ncf_results = evaluator.evaluate_model(
        ncf_trainer, test, n_items, "NCF", user_idx_col="user_idx", use_idx=True,
    )
    for k, v in ncf_results.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k.replace("@", "_at_"), v)
    ncf_trainer.save_model(os.path.join(MODEL_DIR, "ncf_model.pt"))
print("Done!")

# ── Model 5: Hybrid Ensemble ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6/6: Hybrid Ensemble")
print("=" * 60)

with mlflow.start_run(run_name="hybrid_ensemble"):
    hybrid = HybridRecommender(weights=config["models"]["hybrid"]["weights"])
    hybrid.set_mappings(loader.idx_to_item, loader.item_to_idx)
    hybrid.register_model("collaborative", cf_model)
    hybrid.register_model("matrix_factorization", mf_model)
    hybrid.register_model("content_based", cb_model)
    hybrid.register_model("ncf", ncf_trainer)
    mlflow.log_params(hybrid.get_params())

    test_gt = defaultdict(set)
    for _, row in test.iterrows():
        test_gt[row["visitorid"]].add(row["itemid"])

    hybrid_metrics = {
        k: {"precision": [], "recall": [], "ndcg": [], "hit_rate": []}
        for k in config["evaluation"]["top_k"]
    }
    eval_users = list(test_gt.keys())[:500]
    for i, user_id in enumerate(eval_users):
        user_idx = loader.user_to_idx.get(user_id)
        recs = hybrid.predict_for_user(user_id, user_idx, top_n=max(config["evaluation"]["top_k"]))
        rec_items = [item_id for item_id, _ in recs]
        relevant = test_gt[user_id]
        for k in config["evaluation"]["top_k"]:
            hybrid_metrics[k]["precision"].append(precision_at_k(rec_items, relevant, k))
            hybrid_metrics[k]["recall"].append(recall_at_k(rec_items, relevant, k))
            hybrid_metrics[k]["ndcg"].append(ndcg_at_k(rec_items, relevant, k))
            hybrid_metrics[k]["hit_rate"].append(hit_rate_at_k(rec_items, relevant, k))
        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i+1}/{len(eval_users)} users...")

    hybrid_results = {}
    for k in config["evaluation"]["top_k"]:
        for m in ["precision", "recall", "ndcg", "hit_rate"]:
            key = f"{m}@{k}"
            hybrid_results[key] = np.mean(hybrid_metrics[k][m])
            mlflow.log_metric(key.replace("@", "_at_"), hybrid_results[key])
print("Done!")

# ── Final Comparison ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL MODELS TRAINED — COMPARISON")
print("=" * 60)

all_results = {
    "CF-KNN": cf_results,
    "MF-SVD": mf_results,
    "Content-Based": cb_results,
    "NCF": ncf_results,
    "Hybrid": hybrid_results,
}

comparison = evaluator.compare_models(all_results)
print("\n" + comparison.to_string())

print("\n" + "=" * 60)
print("Models saved to: models/")
print("MLflow experiments logged to: mlruns/")
print("Run 'mlflow ui --backend-store-uri mlruns' to view experiments")
print("=" * 60)
