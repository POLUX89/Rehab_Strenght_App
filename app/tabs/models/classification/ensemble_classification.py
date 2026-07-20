"""Sub-rama 'Bagging & Boosting' de Classification (RF, Gradient Boosting, AdaBoost).

render(split) entrena y tunea Random Forest, Gradient Boosting y AdaBoost (con
base DT o SVC) vía GridSearchCV sobre TimeSeriesSplit (CV temporal), scoring F2.
SMOTE opcional dentro del pipeline (solo al train de cada fold, sin fuga);
con SMOTE off, class_weight="balanced" en RF y en las bases de AdaBoost —
Gradient Boosting no soporta class_weight (como KNN en la sub-rama Non Linear).
Reutiliza los helpers de nonlinear_classification. Cachea por (datos, toggle).
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.tabs.models.shap_utils import compute_shap_values

from .nonlinear_classification import _build_pipe, _row, graph_winner, metrics_table


def _synthetic():
    """Render the SMOTE toggle checkbox.

    Returns:
        True if the user enabled SMOTE oversampling, else False.
    """
    return st.checkbox(
        "Balance classes with SMOTE (experimental)",
        False,
        key="ensemble_smote_toggle",
    )


@st.cache_data(show_spinner="Fitting ensemble models... (runs once per dataset)")
def _fit_ensemble(split, synthetic=False):
    """Fit and tune RF, Gradient Boosting and AdaBoost with time-aware CV.

    Builds one pipeline per model ([scaler] → [SMOTE] → estimator), grid-searches
    each over F2 with ``TimeSeriesSplit``, and collects train/test metrics.
    AdaBoost searches over two base estimators (DecisionTreeClassifier and SVC);
    its pipeline includes the scaler because of the SVC branch (trees are
    scale-invariant, so it is harmless for the DT branch). Cached on
    (data, SMOTE toggle).

    Args:
        split: Dict with ``X_train``/``y_train``/``X_test``/``y_test``.
        synthetic: If True, insert SMOTE into every pipeline.

    Returns:
        A ``(results_df, elapsed_seconds, winner_name)`` tuple; ``results_df``
        is indexed by model name and ``winner_name`` is the model with the
        highest test F2 (its fitted estimator lives in
        ``results_df["best_estimator"]``).
    """
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    time0 = time.time()
    f2 = make_scorer(fbeta_score, beta=2)
    tscv = TimeSeriesSplit(n_splits=3)
    rows = []

    # Mismo criterio que logit/nonlinear: con SMOTE off se re-pesa por clase;
    # con SMOTE on no, para no doble-corregir. Gradient Boosting no tiene
    # class_weight (se avisa en render()).
    cw = None if synthetic else "balanced"

    # ---- Random Forest (no scaling needed) ----
    grid_rf = GridSearchCV(
        _build_pipe(
            "rf",
            RandomForestClassifier(random_state=42, class_weight=cw, n_jobs=-1),
            scale=False,
            synthetic=synthetic,
        ),
        param_grid={
            "rf__max_depth": [3, 5, 7, 10],
            "rf__n_estimators": [50, 100, 200],
            "rf__min_samples_leaf": [2, 5, 10],
            # 0.33/0.5/1.0 son fracciones de features (1.0 = todas).
            "rf__max_features": ["sqrt", "log2", 0.33, 0.5, 1.0],
        },
        cv=tscv,
        scoring=f2,
    )
    grid_rf.fit(X_train, y_train)
    rows.append(
        _row(
            "Random Forest",
            grid_rf.best_estimator_,
            grid_rf.best_params_,
            y_train,
            grid_rf.predict(X_train),
            y_test,
            grid_rf.predict(X_test),
        )
    )

    # ---- Gradient Boosting (no scaling needed; no class_weight support) ----
    # `alpha` solo existe en GradientBoostingRegressor (huber/quantile): aquí no aplica.
    grid_gb = GridSearchCV(
        _build_pipe(
            "gb",
            GradientBoostingClassifier(random_state=42),
            scale=False,
            synthetic=synthetic,
        ),
        param_grid={
            "gb__learning_rate": [0.01, 0.1, 0.5],
            "gb__n_estimators": [100, 200],
            "gb__subsample": [0.8, 1.0],
            "gb__min_samples_leaf": [1, 5],
            "gb__max_depth": [2, 3],
            "gb__max_features": ["sqrt", None],
        },
        cv=tscv,
        scoring=f2,
    )
    grid_gb.fit(X_train, y_train)
    rows.append(
        _row(
            "Gradient Boosting",
            grid_gb.best_estimator_,
            grid_gb.best_params_,
            y_train,
            grid_gb.predict(X_train),
            y_test,
            grid_gb.predict(X_test),
        )
    )

    # ---- AdaBoost (scaler in pipe for the SVC base; DT is scale-invariant) ----
    grid_ada = GridSearchCV(
        _build_pipe(
            "ada",
            AdaBoostClassifier(random_state=42),
            scale=True,
            synthetic=synthetic,
        ),
        param_grid=[
            {  # DT base — no scaling needed
                "ada__estimator": [DecisionTreeClassifier(random_state=42, class_weight=cw)],
                "ada__estimator__max_depth": [1, 2, 3],  # DT hyperparameter
                "ada__estimator__min_samples_leaf": [1, 5, 10],  # DT hyperparameter
                "ada__n_estimators": [100, 200, 300],
                "ada__learning_rate": [0.01, 0.1, 0.5],
            },
            {  # SVC base — scaling required
                "ada__estimator": [SVC(random_state=42, class_weight=cw)],
                "ada__estimator__C": [0.1, 1.0, 10.0],  # SVC hyperparameter
                "ada__estimator__kernel": ["rbf", "linear"],  # SVC hyperparameter
                "ada__n_estimators": [100, 200],  # AdaBoost
                "ada__learning_rate": [0.01, 0.1],  # AdaBoost
            },
        ],
        cv=tscv,
        scoring=f2,
    )
    grid_ada.fit(X_train, y_train)
    rows.append(
        _row(
            "AdaBoost",
            grid_ada.best_estimator_,
            grid_ada.best_params_,
            y_train,
            grid_ada.predict(X_train),
            y_test,
            grid_ada.predict(X_test),
        )
    )

    winner = max(rows, key=lambda r: r["test_f2"])["model"]  # name; estimator is in results_df
    results_df = pd.DataFrame(rows).set_index("model")
    return results_df, time.time() - time0, winner


def render(split):
    """Render the Bagging & Boosting sub-branch.

    Reads the SMOTE toggle, fits Random Forest, Gradient Boosting and AdaBoost
    (optional SMOTE), and shows the metrics table, the winner with its
    confusion matrix, the F2 comparison chart and per-model 5×2 metric grids.

    Args:
        split: Dict with ``X_train``/``y_train``/``X_test``/``y_test``.

    Returns:
        None.
    """
    st.title("Bagging & Boosting Models")
    synthetic = _synthetic()  # widget read OUTSIDE the cached function
    if not synthetic:
        st.caption(
            "Imbalance handled with `class_weight='balanced'` (Random Forest and the "
            "AdaBoost base estimators). Gradient Boosting has no `class_weight` — "
            "enable SMOTE to balance it."
        )
    results, elapsed, winner = _fit_ensemble(split, synthetic)
    results = results.sort_values(by="test_f2", ascending=False)
    st.caption(f"Fitted in {elapsed:.2f}s (cached unless data or the SMOTE toggle change).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Performance", divider=True)
        display = results.drop(columns="best_estimator").copy()
        # best_params de AdaBoost contiene objetos estimator: a str para Arrow.
        display["best_params"] = display["best_params"].astype(str)
        st.dataframe(display)
    with c2:
        st.subheader("Model Selection", divider=True)
        st.write(f"🏆 Winner: {winner} — highest F2 on test set")
        best_estimator = results.loc[winner, "best_estimator"]
        cm = confusion_matrix(split["y_test"], best_estimator.predict(split["X_test"]))
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Good", "Bad"],
            yticklabels=["Good", "Bad"],
        )
        ax.set(title=f"Confusion Matrix — {winner} (test)", xlabel="Predicted", ylabel="Actual")
        st.pyplot(fig)
        plt.close(fig)
    graph_winner(results, title="Bagging & Boosting Models")
    metrics_table(results)

    with st.expander("📊 Explanatory Power of Predictors", expanded=True):
        st.subheader(f"📊 Explanatory Power of Predictors — {winner}")
        st.caption(
            "SHAP for the winning model. TreeExplainer for Random Forest / Gradient "
            "Boosting (fast, exact); model-agnostic on the probability of **Bad Sleep** "
            "for AdaBoost."
        )
        # RF/GB → TreeExplainer (positive class); AdaBoost → model-agnostic on
        # predict_proba. cache_key unique per sub-branch (+ synthetic) to avoid the
        # shared-cache collision and to recompute when SMOTE toggles.
        X_background = shap.sample(split["X_train"], 100, random_state=42)
        shap_values = compute_shap_values(
            best_estimator, X_background, split["X_test"], cache_key=("cla_ensemble", synthetic)
        )
        sample_ind = -1  # last sample in the test set

        force_plot = shap.plots.force(shap_values[sample_ind], matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for last sample {split['X_test'].index[sample_ind]}")
        st.pyplot(force_plot)
        plt.close(force_plot)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.bar(shap_values, max_display=14, show=False)
            plt.title("Mean Absolute SHAP Values")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[sample_ind], max_display=14, show=False)
            plt.title(f"SHAP Waterfall Plot for last sample {split['X_test'].index[sample_ind]}")
            st.pyplot(fig)
            plt.close(fig)
