import pandas as pd
import pytest

from rehab_strength.ingest.strong import (
    MAX_PLAUSIBLE_WEIGHT_LBS,
    clean_workouts,
    parse_duration_to_minutes,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1h 5m", 65),
        ("45m", 45),
        ("2h", 120),
        ("01:30:00", 90),
        ("02:30", 2.5),
        ("", None),
    ],
)
def test_parse_duration_to_minutes(raw, expected):
    result = parse_duration_to_minutes(raw)
    if expected is None:
        assert pd.isna(result)
    else:
        assert result == pytest.approx(expected)


def test_parse_duration_handles_na():
    assert pd.isna(parse_duration_to_minutes(None))


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "DATE",
            "WORKOUT_NAME",
            "DURATION_MIN",
            "EXERCISE_NAME",
            "SET_ORDER",
            "WEIGHT_LBS",
            "REPS",
            "DISTANCE",
            "SECONDS",
            "NOTES",
            "WORKOUT_NOTES",
            "RPE",
        ],
    )


def _row(set_order="1", weight=100, exercise="Squat", reps=5):
    return [
        "2026-01-15 10:00:00",
        "Push",
        "1h 5m",
        exercise,
        set_order,
        weight,
        reps,
        None,
        None,
        None,
        None,
        8,
    ]


def test_drops_warmup_and_rest_sets():
    df = clean_workouts(_frame([_row("1"), _row("W"), _row("R"), _row("Rest Timer"), _row("F")]))
    assert len(df) == 1
    assert df["SET_ORDER"].tolist() == ["1"]


def test_drops_implausible_weights():
    df = clean_workouts(_frame([_row(weight=100), _row(weight=MAX_PLAUSIBLE_WEIGHT_LBS + 1)]))
    assert len(df) == 1
    assert df["WEIGHT_LBS"].tolist() == [100]


def test_implausible_weight_does_not_drop_the_whole_workout():
    """Todos los sets de un workout comparten timestamp: solo cae el set malo."""
    df = clean_workouts(
        _frame(
            [
                _row(weight=100, exercise="Squat"),
                _row(weight=MAX_PLAUSIBLE_WEIGHT_LBS + 1, exercise="Bench"),
                _row(weight=150, exercise="Row"),
            ]
        )
    )
    assert df["EXERCISE_NAME"].tolist() == ["Squat", "Row"]


def test_keeps_bodyweight_rows_without_weight():
    df = clean_workouts(_frame([_row(weight=None, exercise="Pushup")]))
    assert len(df) == 1


def test_computes_volume():
    df = clean_workouts(_frame([_row(weight=100, reps=5)]))
    assert df["VOLUME"].iloc[0] == 500


def test_drops_rows_without_exercise_name():
    df = clean_workouts(_frame([_row(), _row(exercise=None)]))
    assert len(df) == 1
