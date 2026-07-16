import numpy as np
import pandas as pd
import pytest

from rehab_strength.ingest.sleep import (
    classify_nap,
    nap_duration,
    nap_status,
    parse_sleep_duration,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10h:30m", 630),
        ("1h 5m", 65),
        ("45m", 45),
        ("2h", 120),
        ("07:30:00", 450),
        ("30:00", 30),
        ("", None),
    ],
)
def test_parse_sleep_duration(raw, expected):
    result = parse_sleep_duration(raw)
    if expected is None:
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (0, 0),  # sin siesta
        (8, -0.2),  # muy temprano
        (11, 0.05),  # temprano
        (14, 0.5),  # ideal
        (18, -0.5),  # tarde
    ],
)
def test_classify_nap_by_start_time(hour, expected):
    assert classify_nap(hour) == expected


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (0, 0),
        (5, 0),  # demasiado corta para contar
        (25, 0.5),  # power nap
        (40, 0.2),
        (55, -0.3),
        (75, 0.1),  # ciclo completo
        (120, -0.4),  # inercia del sueño
    ],
)
def test_nap_duration_score(minutes, expected):
    assert nap_duration(minutes) == expected


def test_nap_scores_handle_na():
    assert classify_nap(np.nan) == 0
    assert nap_duration(np.nan) == 0


def test_nap_status_labels():
    time_scores = pd.Series([0, 0.5, 0.05, 0.05, -0.5])
    duration_scores = pd.Series([0, 0.5, 0.5, 0.0, -0.4])
    result = nap_status(time_scores, duration_scores)
    assert list(result) == ["No Nap", "Boost 🔥", "Good ✅", "Neutral 🤷‍♂️", "Disrupt ⚠️"]
