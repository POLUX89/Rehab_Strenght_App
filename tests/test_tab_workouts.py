"""Tests de la tab Workouts extraída a app/tabs/workouts.py.

Estrategia: matplotlib real (backend Agg, sin display) para ejecutar de verdad
todo el cuerpo de gráficos, mockeando solo Streamlit y plot_line. Así se verifica
que el código movido corre end-to-end, no solo que importa.
"""

from datetime import datetime
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import app.tabs.workouts as wk


def make_st_mock(monkeypatch, chosen_exercise="Squat"):
    m = MagicMock()
    m.selectbox.return_value = chosen_exercise
    m.columns.return_value = (MagicMock(), MagicMock())  # cA, cB
    monkeypatch.setattr(wk, "st", m)
    monkeypatch.setattr(wk, "plot_line", lambda *a, **k: None)
    return m


@pytest.fixture
def workouts_df():
    dates = pd.to_datetime(
        ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
    )
    return pd.DataFrame(
        {
            "Date": dates,
            "DATE": dates,
            "EXERCISE_NAME": ["Squat"] * 4 + ["Bench"] * 2,
            "est_1RM": [100, 110, 120, 130, 80, 85],
            "VOLUME": [1000, 1100, 1200, 1300, 800, 850],
            "RPE": [7, 8, 8, 9, 6, 7],
            "WEIGHT_LBS": [80, 88, 96, 104, 64, 68],
            "REPS": [5, 5, 5, 5, 5, 5],
        }
    )


def test_render_none_infos(monkeypatch):
    m = make_st_mock(monkeypatch)
    wk.render(None, datetime(2025, 5, 14), 15)
    m.info.assert_called_once()


def test_render_missing_columns_errors(monkeypatch):
    m = make_st_mock(monkeypatch)
    wk.render(pd.DataFrame({"foo": [1]}), datetime(2025, 5, 14), 15)
    m.error.assert_called_once()


def test_render_full_runs_without_error(monkeypatch, workouts_df):
    # con datos completos, el cuerpo (bar, líneas, MA, RPE, volumen) debe correr entero
    make_st_mock(monkeypatch, chosen_exercise="Squat")
    wk.render(workouts_df, datetime(2025, 5, 14), 15)


def test_render_mutates_week_column_in_place(monkeypatch, workouts_df):
    # la mutación in-place de 'Week' debe reflejarse en el df pasado (contrato por referencia)
    make_st_mock(monkeypatch, chosen_exercise="Squat")
    wk.render(workouts_df, datetime(2025, 5, 14), 15)
    assert "Week" in workouts_df.columns
