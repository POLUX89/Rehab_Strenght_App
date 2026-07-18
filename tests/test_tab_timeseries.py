"""Tests de la tab Time Series extraída a app/tabs/timeseries.py.

matplotlib real (Agg) + statsmodels real (ADF/KPSS/ACF/PACF corren de verdad),
mockeando solo Streamlit. Verifica que render devuelve el veredicto que la tab
Models reutiliza.
"""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.timeseries as ts

TSA_COLS = [
    "Start",
    "End",
    "InBed hrs",
    "Asleep hrs",
    "Awake",
    "REM hrs",
    "Light hrs",
    "Deep hrs",
    "Efficiency",
    "Fall Asleep",
]


@pytest.fixture
def recovery_df():
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    rng = np.random.default_rng(0)
    data = {"Date": dates, "Score": rng.normal(80, 5, 90)}
    for c in TSA_COLS:
        data[c] = 0  # columnas requeridas por tsa_col; solo Score se usa
    return pd.DataFrame(data)


def test_render_returns_valid_verdict(monkeypatch, recovery_df):
    m = MagicMock()
    m.slider.return_value = 20
    monkeypatch.setattr(ts, "st", m)
    result = ts.render(recovery_df)
    assert result in {"Stationary", "Non-Stationary", "Inconclusive"}


def test_render_writes_overall_verdict(monkeypatch, recovery_df):
    m = MagicMock()
    m.slider.return_value = 20
    monkeypatch.setattr(ts, "st", m)
    result = ts.render(recovery_df)
    # el veredicto se escribe con st.write en la propia tab
    written = " ".join(str(c.args[0]) for c in m.write.call_args_list if c.args)
    assert result in written
