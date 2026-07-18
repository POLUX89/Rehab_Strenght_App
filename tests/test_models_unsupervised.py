"""Tests del sub-tab Unsupervised de Models (app/tabs/models/unsupervised.py)."""

import inspect
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.models.unsupervised as unsup


def test_render_signature():
    assert list(inspect.signature(unsup.render).parameters) == ["df_model"]


@pytest.fixture
def df_model():
    n = 60
    rng = np.random.default_rng(0)
    cols = {
        "REM hrs": rng.normal(1.5, 0.3, n),
        "Stress_prev_day": rng.normal(30, 5, n),
        "Deep hrs": rng.normal(1.2, 0.2, n),
        "Wake Count": rng.integers(0, 4, n),
        "Sleep_hr_surplus": rng.normal(0, 1, n),
        "Respiration": rng.normal(14, 1, n),
        "Stress_sleep": rng.normal(25, 5, n),
        "Score": rng.normal(80, 6, n),
    }
    return pd.DataFrame(cols)


@pytest.mark.parametrize("choice", ["PCA", "T-SNE", "K-Means"])
def test_render_runs_each_technique(monkeypatch, df_model, choice):
    m = MagicMock()
    m.selectbox.return_value = choice
    m.slider.return_value = 3
    m.columns.side_effect = lambda spec, *a, **k: tuple(
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    )
    monkeypatch.setattr(unsup, "st", m)
    unsup.render(df_model.copy())  # no debe lanzar
