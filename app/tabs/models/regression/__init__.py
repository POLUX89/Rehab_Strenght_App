"""Regression — despachador del tipo de modelo.

render(df_model, predictors) muestra el segmented_control de tipo de modelo y
delega en las sub-ramas: ols, linear (Other Linear), nonlinear (Non Linear) y
ensemble (Bagging & Boosting). Cada sub-rama es autónoma y recibe (df_model,
predictors).
"""

from __future__ import annotations

import streamlit as st

from . import ensemble, linear, nonlinear, ols


def render(df_model, predictors):
    models = st.segmented_control(
        "Select Model Type:",
        [
            "OLS diagnosis",
            "Other Linear Models",
            "Non Linear Models",
            "Bagging & Boosting Models",
        ],
        key="model_type_control",
        default="OLS diagnosis",
    )
    if models == "OLS diagnosis":
        ols.render(df_model, predictors)
    elif models == "Other Linear Models":
        linear.render(df_model, predictors)
    elif models == "Non Linear Models":
        nonlinear.render(df_model, predictors)
    elif models == "Bagging & Boosting Models":
        ensemble.render(df_model, predictors)
