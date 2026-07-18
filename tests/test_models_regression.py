"""Tests del sub-paquete Regression de Models (app/tabs/models/regression/).

La garantía fuerte de la subdivisión es la comparación AST (todas las llamadas
del original preservadas, solo se añaden las 4 delegaciones), hecha fuera de
pytest. Aquí: firmas y que el despachador enruta cada opción del segmented_control
a su sub-rama.
"""

import inspect
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.models.regression as regression
from app.tabs.models.regression import ensemble, linear, nonlinear, ols

PREDICTORS = [
    "REM hrs",
    "Stress_prev_day",
    "Deep hrs",
    "Wake Count",
    "Sleep_hr_surplus",
    "Respiration",
    "Stress_sleep",
]


@pytest.fixture
def df_model():
    n = 60
    rng = np.random.default_rng(0)
    data = {"Date": pd.date_range("2026-01-01", periods=n, freq="D")}
    for c in PREDICTORS:
        data[c] = rng.normal(0, 1, n)
    data["Score"] = rng.normal(80, 6, n)
    data["Quality"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_all_renders_share_signature():
    for mod in (regression, ols, linear, nonlinear, ensemble):
        assert list(inspect.signature(mod.render).parameters) == ["df_model", "predictors"]


@pytest.mark.parametrize(
    ("choice", "target"),
    [
        ("OLS diagnosis", "ols"),
        ("Other Linear Models", "linear"),
        ("Non Linear Models", "nonlinear"),
        ("Bagging & Boosting Models", "ensemble"),
    ],
)
def test_dispatch_routes_to_correct_branch(monkeypatch, df_model, choice, target):
    m = MagicMock()
    m.segmented_control.return_value = choice
    monkeypatch.setattr(regression, "st", m)
    called = []
    for name, mod in [
        ("ols", ols),
        ("linear", linear),
        ("nonlinear", nonlinear),
        ("ensemble", ensemble),
    ]:
        monkeypatch.setattr(mod, "render", lambda df, p, _n=name: called.append(_n))
    regression.render(df_model, PREDICTORS)
    assert called == [target]  # solo la rama elegida se ejecuta


def test_dispatch_none_runs_nothing(monkeypatch, df_model):
    m = MagicMock()
    m.segmented_control.return_value = None
    monkeypatch.setattr(regression, "st", m)
    for mod in (ols, linear, nonlinear, ensemble):
        monkeypatch.setattr(mod, "render", lambda df, p: pytest.fail("no debería entrenar"))
    regression.render(df_model, PREDICTORS)  # ninguna rama
