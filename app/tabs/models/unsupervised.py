"""Sub-tab Unsupervised de Models (PCA / T-SNE / K-Means / DBSCAN).

Extraída de models/__init__.py sin cambiar la lógica. Solo necesita df_model.
Los imports de sklearn que estaban dentro de las ramas se conservan locales.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler


def render(df_model):
    from sklearn.decomposition import PCA

    unsupervised_options = ["PCA", "T-SNE", "K-Means", "DBSCAN"]
    unsupervised_choice = st.selectbox(
        "Select unsupervised technique:",
        options=unsupervised_options,
        key="unsupervised_choice",
    )
    if unsupervised_choice == "PCA":
        st.header("📊 Principal Component Analysis (PCA)")
        df_model["Bad_Sleep"] = (df_model["Score"] < 80).astype(int)
        predictors = [
            "REM hrs",
            "Stress_prev_day",
            "Deep hrs",
            "Wake Count",
            "Sleep_hr_surplus",
            "Respiration",
            "Stress_sleep",
        ]
        X_scaled = StandardScaler().fit_transform(df_model[predictors])
        best_pca = PCA(n_components=0.85)
        X_pca = best_pca.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            ax=ax,
            hue=df_model["Bad_Sleep"],
            palette=["green", "red"],
            alpha=0.7,
        )
        ax.set_title("PCA: First Two Principal Components")
        sns.despine(ax=ax)
        st.pyplot(fig)

        num_components = [1, 2, 3, 4, 5, 6, 7]
        results = []
        for i in num_components:
            pca = PCA(n_components=i, random_state=42)
            pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_.sum() * 100
            explained_var_ = pca.explained_variance_ratio_
            results.append(
                {
                    "Components": i,
                    "Explained Variance (%)": explained_variance,
                    "Explained Variance": explained_var_,
                }
            )

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        # PVE Proportion of Variance Explained

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        results_df.plot(
            kind="bar",
            x="Components",
            y="Explained Variance (%)",
            color=sns.color_palette("icefire", n_colors=len(results_df)),
            legend=False,
            ax=ax[1],
        )
        bars = plt.gca().patches
        values = results_df["Explained Variance (%)"].values
        for bar, value in zip(bars, values, strict=False):
            ax[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax[1].set_title(
            "Explained Variance by Number of PCA Components",
            fontsize=10,
            fontweight="bold",
            pad=15,
        )
        ax[1].set_xlabel("Number of Components")
        ax[1].set_ylabel("Explained Variance (%)")
        ax[1].set_ylim(0, 100)
        sns.despine(ax=ax[1])
        plt.tight_layout()

        pca_df = pd.DataFrame(
            best_pca.components_,
            columns=predictors,
            index=[f"PC{i + 1}" for i in range(best_pca.n_components_)],
        )
        sns.heatmap(
            pca_df,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Loading"},
            vmin=-1,
            vmax=1,
            ax=ax[0],
        )
        ax[0].set_title(
            "PCA Loadings — each cell shows feature contribution to each PC",
            fontweight="bold",
            fontsize=10,
            pad=15,
        )
        plt.tight_layout()
        st.pyplot(fig)
    elif unsupervised_choice == "T-SNE":
        st.header("📊 T-Distributed Stochastic Neighbor Embedding (T-SNE)")
        from sklearn.manifold import TSNE

        df_model["Bad_Sleep"] = (df_model["Score"] < 80).astype(int)
        predictors = [
            "REM hrs",
            "Stress_prev_day",
            "Deep hrs",
            "Wake Count",
            "Sleep_hr_surplus",
            "Respiration",
            "Stress_sleep",
        ]
        X_scaled = StandardScaler().fit_transform(df_model[predictors])
        tsne = TSNE(n_components=2, perplexity=10)
        X_tsne = tsne.fit_transform(X_scaled)
        pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            ax=ax,
            hue=df_model["Bad_Sleep"].map({0: "Good Sleep", 1: "Bad Sleep"}),
            palette=["green", "red"],
            alpha=0.7,
        )
        ax.set_title("T-SNE: 2D Visualization of Sleep Data")
        sns.despine(ax=ax)
        st.pyplot(fig)

    elif unsupervised_choice == "K-Means":
        st.header("📊 K-Means")
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score, silhouette_score

        predictors = [
            "REM hrs",
            "Stress_prev_day",
            "Deep hrs",
            "Wake Count",
            "Sleep_hr_surplus",
            "Respiration",
            "Stress_sleep",
        ]
        df_model["Bad_Sleep"] = (df_model["Score"] < 80).astype(int)
        X_scaled = StandardScaler().fit_transform(df_model[predictors])

        # Train K-Means for K=2 to 10 and evaluate with Inertia, Silhouette Score, and Davies-Bouldin Score
        results_cluster = []
        for k in range(2, 11):  # K from 2 to 10
            kmeans = KMeans(n_clusters=k, random_state=42, algorithm="elkan")
            kmeans.fit(X_scaled)
            results_cluster.append(
                {
                    "K": k,
                    "Inertia": kmeans.inertia_,
                    "Silhouette Score": silhouette_score(X_scaled, kmeans.labels_),
                    "Davies-Bouldin Score": davies_bouldin_score(X_scaled, kmeans.labels_),
                }
            )
        results_cluster_df = pd.DataFrame(results_cluster).sort_values(
            by="Silhouette Score", ascending=False
        )
        st.dataframe(results_cluster_df)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        plt.suptitle("K-Means Clustering Evaluation Metrics vs K", fontsize=16, fontweight="bold")
        sns.lineplot(
            x="K", y="Inertia", data=results_cluster_df, marker="o", color="blue", ax=axes[0]
        )
        axes[0].set_title("K-Means Inertia vs K", fontsize=8, fontweight="bold")
        axes[0].set_xlabel("Number of Clusters (K)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_xticks(results_cluster_df["K"])
        axes[0].grid()
        sns.despine(ax=axes[0])

        sns.lineplot(
            x="K",
            y="Silhouette Score",
            data=results_cluster_df,
            marker="o",
            color="orange",
            ax=axes[1],
        )
        axes[1].set_title("K-Means Silhouette Score vs K", fontsize=8, fontweight="bold")
        axes[1].set_xlabel("Number of Clusters (K)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_xticks(results_cluster_df["K"])
        axes[1].grid()
        sns.despine(ax=axes[1])

        sns.lineplot(
            x="K",
            y="Davies-Bouldin Score",
            data=results_cluster_df,
            marker="o",
            color="green",
            ax=axes[2],
        )
        axes[2].set_title("K-Means Davies-Bouldin Score vs K", fontsize=8, fontweight="bold")
        axes[2].set_xlabel("Number of Clusters (K)")
        axes[2].set_ylabel("Davies-Bouldin Score")
        axes[2].set_xticks(results_cluster_df["K"])
        axes[2].grid()
        sns.despine(ax=axes[2])

        plt.tight_layout()
        st.pyplot(fig)

        # --------------------------------------
        # Select best K and fit the K-Means model
        best_k = results_cluster_df.iloc[0]["K"]
        kmeans = KMeans(n_clusters=int(best_k), random_state=42, algorithm="elkan")
        kmeans.fit(X_scaled)
        df_model["KMeans_Cluster"] = kmeans.labels_

        st.subheader(f"📊 Cluster Analysis for K={int(best_k)}")
        cluster_analysis = (
            df_model.groupby("KMeans_Cluster")
            .agg(
                {
                    "Score": "mean",
                    "REM hrs": "mean",
                    "Stress_prev_day": "mean",
                    "Deep hrs": "mean",
                    "Wake Count": "mean",
                    "Sleep_hr_surplus": "mean",
                    "Respiration": "mean",
                    "Stress_sleep": "mean",
                }
            )
            .round(2)
        )

        st.dataframe(cluster_analysis)

        st.subheader("📊 T-Distributed Stochastic Neighbor Embedding (T-SNE)")
        from sklearn.manifold import TSNE

        predictors = [
            "REM hrs",
            "Stress_prev_day",
            "Deep hrs",
            "Wake Count",
            "Sleep_hr_surplus",
            "Respiration",
            "Stress_sleep",
        ]
        X_scaled = StandardScaler().fit_transform(df_model[predictors])
        tsne = TSNE(n_components=2, perplexity=10)
        X_tsne = tsne.fit_transform(X_scaled)
        pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            ax=ax,
            hue=df_model["KMeans_Cluster"],
            palette="viridis",
            alpha=0.7,
        )
        ax.set_title("T-SNE: 2D Visualization of Sleep Data")
        sns.despine(ax=ax)
        st.pyplot(fig)
