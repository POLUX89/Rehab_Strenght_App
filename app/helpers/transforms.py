"""Helpers de transformación de datos puros (sin Streamlit).

Extraídos de streamlit_app.py sin cambiar la lógica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_unique_columns(cols):
    """Deduplicate column names by suffixing repeats with ``.1``, ``.2``, ...

    Streamlit CSV uploads can preserve duplicate headers; this makes them
    unique so downstream selection by name is unambiguous.

    Args:
        cols: Iterable of column names (coerced to stripped strings).

    Returns:
        A list of unique column names, in the original order.
    """
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out


def pick_col(df, candidates):
    """Return the first candidate column that exists in the DataFrame.

    Args:
        df: DataFrame to inspect, or None.
        candidates: Ordered iterable of column names to try.

    Returns:
        The first name in ``candidates`` present in ``df.columns``, or None
        if ``df`` is None or none of the candidates match.
    """
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_date(df, col="Date"):
    """Parse a column to datetime floored to day resolution, in place.

    Unparseable values become ``NaT``. No-op if the column is absent.

    Args:
        df: DataFrame to modify in place.
        col: Name of the date column. Defaults to ``"Date"``.

    Returns:
        The same DataFrame, with ``col`` coerced to day-floored datetimes.
    """
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("D")
    return df


def epley_1rm(w, r):
    """Estimate a one-rep max with the Epley formula ``w * (1 + r / 30)``.

    Args:
        w: Weight lifted.
        r: Repetitions performed.

    Returns:
        The estimated 1RM as a float, or ``np.nan`` if either input is
        missing or cannot be converted to a number.
    """
    try:
        w = float(w)
        r = float(r)
        if np.isnan(w) or np.isnan(r):
            return np.nan
        return w * (1.0 + r / 30.0)
    except Exception:
        return np.nan


def safe_numeric(s):
    """Coerce a Series to numeric, turning invalid values into ``NaN``.

    Args:
        s: Series (or array-like) to convert.

    Returns:
        The values parsed with ``pd.to_numeric(errors="coerce")``.
    """
    return pd.to_numeric(s, errors="coerce")


def daily_ma(series, window_days):
    """Compute a rolling mean over a daily-indexed series.

    Uses ``min_periods = max(1, window_days // 2)`` so partially filled
    windows still produce a value.

    Args:
        series: Series indexed by day (or already resampled to daily).
        window_days: Rolling window size, in days.

    Returns:
        The rolling-mean Series.
    """
    return series.rolling(window=window_days, min_periods=max(1, window_days // 2)).mean()


def weekly_bucket(dt_series):
    """Map each date to the start (Monday) of its ISO week.

    Args:
        dt_series: Date-like Series; invalid values are coerced to ``NaT``.

    Returns:
        A Series of week-start timestamps (weeks anchored on Monday).
    """
    dt = pd.to_datetime(dt_series, errors="coerce")  # coerce bad strings to NaT
    return dt.dt.to_period("W-MON").dt.start_time


def week_bounds(today=None):
    """Return the Monday and Sunday bounding the week of a given day.

    Args:
        today: Reference date. Defaults to the current day when None.

    Returns:
        A ``(start, end)`` tuple of timestamps for Monday and Sunday of that
        week.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()
    start = today - pd.Timedelta(days=today.weekday())
    end = start + pd.Timedelta(days=6)
    return start, end


def safe_minimal_last(df, date_col, value_col):
    """Return the most recent non-null value of a column, ordered by date.

    Args:
        df: Source DataFrame, or None.
        date_col: Column used to order rows chronologically.
        value_col: Column whose latest value is returned, or None.

    Returns:
        The last value of ``value_col`` after dropping missing rows and
        sorting by ``date_col``, or None if inputs are missing, the columns
        are absent, or no valid rows remain.
    """
    if df is None or value_col is None:
        return None
    if date_col not in df.columns or value_col not in df.columns:
        return None
    tmp = df[[date_col, value_col]].copy()
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if tmp.empty:
        return None
    return tmp[value_col].iloc[-1]


def recovery_zone(x):
    """Classify a normalized recovery score into a labelled zone.

    Args:
        x: Recovery score in the 0-1 range, or None/NaN.

    Returns:
        A label with emoji: ``"🟢 ⬆️ Ready"`` for ``x >= 0.7``,
        ``"🟡 Moderate"`` for ``x >= 0.55``, ``"🔴 ⬇️ Low"`` otherwise, or
        ``"No data"`` when ``x`` is missing.
    """
    if x is None or pd.isna(x):
        return "No data"
    if x >= 0.7:
        return "🟢 ⬆️ Ready"
    if x >= 0.55:
        return "🟡 Moderate"
    return "🔴 ⬇️ Low"


def sleep_classifier(q):
    """Map a qualitative sleep rating to a binary label.

    Args:
        q: Sleep quality label (e.g. ``"Good"``, ``"Excellent"``, ``"Poor"``).

    Returns:
        ``1`` if the rating is ``"Good"`` or ``"Excellent"``, else ``0``.
    """
    return 1 if q in ["Good", "Excellent"] else 0


def string_to_decimal_hours(time_str):
    """Parse a ``"Xh Ymin"`` duration string into decimal hours.

    Handles the hours-and-minutes, hours-only and minutes-only forms.

    Args:
        time_str: Duration string such as ``"7h 30min"``, ``"8h"`` or
            ``"45min"``.

    Returns:
        The duration in decimal hours as a float, or ``np.nan`` if the input
        is missing or matches none of the expected forms.
    """
    if pd.isna(time_str):
        return np.nan
    time_str = time_str.strip()
    if "h" in time_str and "min" in time_str:
        hours, minutes = time_str.split("h")
        minutes = minutes.replace("min", "").strip()
        return float(hours.strip()) + float(minutes) / 60
    elif "h" in time_str:
        hours = time_str.replace("h", "").strip()
        return float(hours)
    elif "min" in time_str:
        minutes = time_str.replace("min", "").strip()
        return float(minutes) / 60
    else:
        return np.nan


def normalize_workouts(df):
    """Standardize a raw Strong workouts DataFrame in place.

    Parses ``DATE`` to datetime, trims exercise names, coerces the numeric
    columns, derives ``est_1RM`` (Epley) when weight and reps are present,
    and adds day-floored ``Date``/``DAY`` helper columns.

    Args:
        df: Raw workouts DataFrame.

    Returns:
        The same DataFrame with normalized and derived columns.
    """
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    if "EXERCISE_NAME" in df.columns:
        df["EXERCISE_NAME"] = df["EXERCISE_NAME"].astype(str).str.strip()
    for col in ["WEIGHT_LBS", "REPS", "RPE", "VOLUME"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if set(["WEIGHT_LBS", "REPS"]).issubset(df.columns):
        df["est_1RM"] = df.apply(lambda r: epley_1rm(r["WEIGHT_LBS"], r["REPS"]), axis=1)

    if "DATE" in df.columns:
        df["Date"] = df["DATE"].dt.floor("D")
        df["DAY"] = df["Date"]
    return df


def normalize_sleep(df):
    """Standardize a raw sleep DataFrame in place.

    Deduplicates headers, renames the first date-like column to ``Date``,
    floors it to day resolution, and coerces the known sleep metric columns
    to numeric.

    Args:
        df: Raw sleep DataFrame.

    Returns:
        The normalized DataFrame.
    """
    df.columns = make_unique_columns(df.columns)
    if "Date" not in df.columns:
        for cand in ["DATE", "day", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    df = coerce_date(df, "Date")
    for cand in [
        "Score",
        "Wake Count",
        "Efficiency",
        "Asleep hrs",
        "InBed hrs",
        "REM hrs",
        "Light hrs",
        "Deep hrs",
    ]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df


def normalize_recovery(df):
    """Standardize a raw recovery DataFrame in place.

    Deduplicates headers, renames the first date-like column to ``Date``,
    floors it to day resolution, and coerces the known recovery metric
    columns to numeric.

    Args:
        df: Raw recovery DataFrame.

    Returns:
        The normalized DataFrame.
    """
    df.columns = make_unique_columns(df.columns)
    if "Date" not in df.columns:
        for cand in ["DATE", "day", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    df = coerce_date(df, "Date")
    for cand in [
        "Sigmoid Recovery Score",
        "RECOVERY_SCORE_RAW",
        "Stress_prev_day",
        "Overnight HRV",
        "Resting Heart Rate",
        "Score",
    ]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df
