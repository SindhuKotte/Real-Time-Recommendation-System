"""
Streamlit Dashboard for the E-Commerce Recommendation System.

Features:
- Interactive user recommendation explorer
- Model performance comparison
- Item similarity browser
- User behavior analytics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.recommender import RecommendationEngine
from src.utils import load_config

st.set_page_config(
    page_title="E-Commerce Recommendation Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_engine():
    """Load the recommendation engine (cached)."""
    engine = RecommendationEngine(config_path="configs/config.yaml")
    try:
        engine.load_data()
        engine.load_models()
    except Exception as e:
        st.warning(f"Models not loaded: {e}. Run training notebook first.")
    return engine


engine = load_engine()

st.title("E-Commerce Real-Time Recommendation Engine")
st.markdown("Interactive dashboard for exploring product recommendations across multiple ML models.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Recommendations", "Model Comparison", "Item Explorer", "Data Insights"
])


# ─── Tab 1: Recommendations ─────────────────────────────────────────
with tab1:
    st.header("Personalized Recommendations")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if engine.interactions is not None:
            sample_users = engine.interactions["visitorid"].unique()[:1000].tolist()
            user_id = st.selectbox("Select User ID", options=sample_users)
        else:
            user_id = st.number_input("Enter User ID", min_value=0, value=0)

    with col2:
        strategies = engine.get_available_strategies()
        strategy = st.selectbox("Strategy", options=strategies, index=0)

    with col3:
        top_n = st.slider("Number of Recommendations", 5, 50, 10)

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recs = engine.recommend(user_id=user_id, top_n=top_n, strategy=strategy)

        if recs:
            st.success(f"Generated {len(recs)} recommendations using **{strategy}** strategy")

            rec_df = pd.DataFrame(recs)
            st.dataframe(rec_df, use_container_width=True)

            fig = px.bar(
                rec_df, x="item_id", y="score", color="score",
                color_continuous_scale="Blues",
                title=f"Top-{top_n} Recommendations for User {user_id}",
                labels={"item_id": "Item ID", "score": "Relevance Score"},
            )
            fig.update_layout(xaxis_type="category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No recommendations generated. Try a different strategy.")

    st.subheader("User Interaction History")
    history = engine.get_user_history(user_id)
    if history:
        hist_df = pd.DataFrame(history)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Items Interacted", len(history))
            st.dataframe(hist_df.head(20), use_container_width=True)
        with col_b:
            fig = px.histogram(
                hist_df, x="rating", nbins=20,
                title="User's Rating Distribution",
                color_discrete_sequence=["#3498db"]
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No history available for this user.")


# ─── Tab 2: Model Comparison ────────────────────────────────────────
with tab2:
    st.header("Model Performance Comparison")

    st.markdown("""
    Compare recommendations across all available strategies for the same user.
    This helps understand the diversity and overlap between models.
    """)

    if engine.interactions is not None:
        comp_user = st.selectbox(
            "Select User for Comparison",
            options=engine.interactions["visitorid"].unique()[:500].tolist(),
            key="comp_user"
        )

        if st.button("Compare All Models", type="primary"):
            all_model_recs = {}

            for strat in strategies:
                recs = engine.recommend(user_id=comp_user, top_n=10, strategy=strat)
                if recs:
                    all_model_recs[strat] = recs

            if all_model_recs:
                fig = make_subplots(
                    rows=1, cols=len(all_model_recs),
                    subplot_titles=list(all_model_recs.keys())
                )

                for i, (strat_name, recs) in enumerate(all_model_recs.items(), 1):
                    df = pd.DataFrame(recs)
                    fig.add_trace(
                        go.Bar(
                            x=df["item_id"].astype(str),
                            y=df["score"],
                            name=strat_name,
                            marker_color=px.colors.qualitative.Set2[i % 8]
                        ),
                        row=1, col=i
                    )

                fig.update_layout(
                    height=400,
                    title_text=f"Recommendations Comparison — User {comp_user}",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Overlap Analysis")
                item_sets = {
                    name: set(r["item_id"] for r in recs)
                    for name, recs in all_model_recs.items()
                }
                overlap_data = []
                names = list(item_sets.keys())
                for i in range(len(names)):
                    for j in range(len(names)):
                        set_i = item_sets[names[i]]
                        set_j = item_sets[names[j]]
                        overlap = len(set_i & set_j) / max(len(set_i | set_j), 1)
                        overlap_data.append({
                            "Model A": names[i], "Model B": names[j],
                            "Jaccard Overlap": round(overlap, 3)
                        })

                overlap_df = pd.DataFrame(overlap_data).pivot(
                    index="Model A", columns="Model B", values="Jaccard Overlap"
                )
                fig_heat = px.imshow(
                    overlap_df, text_auto=".2f",
                    color_continuous_scale="RdYlBu_r",
                    title="Model Overlap (Jaccard Similarity)"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Load data first by running the preprocessing pipeline.")


# ─── Tab 3: Item Explorer ───────────────────────────────────────────
with tab3:
    st.header("Item Similarity Explorer")

    if engine.interactions is not None:
        sample_items = engine.interactions["itemid"].unique()[:500].tolist()
        item_id = st.selectbox("Select Item ID", options=sample_items)

        n_similar = st.slider("Number of Similar Items", 5, 30, 10, key="n_sim")

        if st.button("Find Similar Items", type="primary"):
            similar = engine.get_similar_items(item_id, top_n=n_similar)

            if similar:
                sim_df = pd.DataFrame(similar)
                st.dataframe(sim_df, use_container_width=True)

                fig = px.bar(
                    sim_df, x="item_id", y="score",
                    color="score", color_continuous_scale="Greens",
                    title=f"Items Similar to {item_id}",
                    labels={"score": "Similarity Score"}
                )
                fig.update_layout(xaxis_type="category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No similar items found. Ensure content-based model is trained.")
    else:
        st.info("Load data first.")


# ─── Tab 4: Data Insights ───────────────────────────────────────────
with tab4:
    st.header("Dataset Analytics")

    if engine.interactions is not None:
        df = engine.interactions

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Interactions", f"{len(df):,}")
        col2.metric("Unique Users", f"{df['visitorid'].nunique():,}")
        col3.metric("Unique Items", f"{df['itemid'].nunique():,}")
        sparsity = 1 - len(df) / (df["visitorid"].nunique() * df["itemid"].nunique())
        col4.metric("Sparsity", f"{sparsity:.2%}")

        col_a, col_b = st.columns(2)

        with col_a:
            user_counts = df.groupby("visitorid").size().reset_index(name="interactions")
            fig = px.histogram(
                user_counts, x="interactions", nbins=50,
                title="User Activity Distribution",
                labels={"interactions": "# Interactions", "count": "# Users"},
                color_discrete_sequence=["#3498db"]
            )
            fig.update_xaxes(range=[0, user_counts["interactions"].quantile(0.95)])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            item_counts = df.groupby("itemid").size().reset_index(name="interactions")
            fig = px.histogram(
                item_counts, x="interactions", nbins=50,
                title="Item Popularity Distribution",
                labels={"interactions": "# Interactions", "count": "# Items"},
                color_discrete_sequence=["#e74c3c"]
            )
            fig.update_xaxes(range=[0, item_counts["interactions"].quantile(0.95)])
            st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(
            df, x="rating", nbins=50,
            title="Implicit Rating Distribution",
            color_discrete_sequence=["#2ecc71"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data loaded. Run the preprocessing pipeline first.")


st.markdown("---")
st.markdown(
    "**E-Commerce Real-Time Recommendation System** | "
    "Built with PyTorch, FastAPI, Streamlit & MLflow"
)
