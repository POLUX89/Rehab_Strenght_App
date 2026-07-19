"""Ingesta de sueño (Google Sheets) + HRV y Garmin (Excel) -> sueño y recovery limpios.

Produce dos datasets:
  - clean_sleep_data.csv    sueño principal + naps + HRV
  - clean_recovery_data.csv lo anterior + Garmin y el Sigmoid Recovery Score
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    CLEAN_RECOVERY_CSV,
    CLEAN_SLEEP_CSV,
    GARMIN_SLEEP_XLSX,
    HEALTH_METRICS_PREFIX,
    HRV_XLSX,
    SLEEP_TAB,
    ensure_dirs,
)

# Columnas de la hoja que no se usan aguas abajo.
DROP_COLUMNS = [
    "Min. Respiration Rate",
    "Max. Respiration Rate",
    "Avg. Respiration Rate",
    "Wrist Temperature",
    "Low SpO2",
    "High SpO2",
    "Avg. SpO2",
    "Low HRV",
    "High HRV",
    "Avg. HRV",
]

DURATION_COLUMNS = ["InBed", "Asleep", "Awake", "REM", "Light", "Deep", "Fall Asleep"]

# Componentes del recovery score. Todos pesan igual.
RECOVERY_COMPONENTS = [
    "Asleep hrs",
    "Overnight HRV",
    "Resting Heart Rate",
    "Score",
    "Stress_prev_day",
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_sleep_duration(value) -> float:
    """Parse a sleep-duration string into total minutes.

    Handles the ``"10h:30m"``, ``"HH:MM:SS"``, ``"MM:SS"``, ``"45m"`` and
    ``"2h"`` forms.

    Args:
        value: Duration value; may be a string or missing.

    Returns:
        The duration in minutes as a float, or ``np.nan`` if the value is
        missing or empty.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan

    if ":" in s and ("h" in s or "m" in s):
        hours = minutes = 0
        for part in s.split(":"):
            if "h" in part:
                hours = int(part.replace("h", ""))
            elif "m" in part:
                minutes = int(part.replace("m", ""))
        return hours * 60 + minutes

    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = (int(p) for p in parts)
            return h * 60 + m + sec / 60
        if len(parts) == 2:
            m, sec = (int(p) for p in parts)
            return m + sec / 60

    hours = minutes = 0
    if h_match := re.search(r"(\d+)\s*h", s.lower()):
        hours = int(h_match.group(1))
    if m_match := re.search(r"(\d+)\s*m", s.lower()):
        minutes = int(m_match.group(1))
    return hours * 60 + minutes


# ---------------------------------------------------------------------------
# Scoring de naps
# ---------------------------------------------------------------------------
def classify_nap(time_decimal: float) -> float:
    """Score a nap by its start time of day.

    Rewards early-afternoon naps and penalizes very early or late ones.

    Args:
        time_decimal: Nap start time as decimal hours (e.g. 14.5 for 14:30).

    Returns:
        A score in ``[-0.5, 0.5]``, or ``0`` when the time is missing or zero.
    """
    if pd.isna(time_decimal) or time_decimal == 0:
        return 0
    if time_decimal < 10:
        return -0.2  # muy temprano
    if time_decimal < 13:
        return 0.05  # temprano
    if time_decimal < 16:
        return 0.5  # ideal
    return -0.5  # tarde


def nap_duration(duration: float) -> float:
    """Score a nap by its length in minutes.

    Favors short restorative naps (up to ~30 min) and penalizes lengths
    associated with sleep inertia.

    Args:
        duration: Nap duration in minutes.

    Returns:
        A score in ``[-0.4, 0.5]``, or ``0`` when the duration is missing,
        zero, or under 10 minutes.
    """
    if pd.isna(duration) or duration == 0:
        return 0
    if duration < 10:
        return 0
    if duration <= 30:
        return 0.5
    if duration <= 45:
        return 0.2
    if duration <= 60:
        return -0.3
    if duration <= 90:
        return 0.1
    return -0.4


def nap_status(nap_time_score, nap_duration_score):
    """Label the net effect of a nap from its time and duration scores.

    Categories by combined score: ``No Nap`` (both scores zero), ``Boost``
    (clearly beneficial), ``Good`` (moderately positive), ``Neutral`` (small
    or unclear effect), ``Disrupt`` (likely harms night sleep or causes
    inertia).

    Args:
        nap_time_score: Score from :func:`classify_nap`.
        nap_duration_score: Score from :func:`nap_duration`.

    Returns:
        A NumPy array of label strings (one per row), aligned with the input
        series.
    """
    total = nap_time_score + nap_duration_score
    conditions = [
        (nap_time_score == 0) & (nap_duration_score == 0),
        total >= 0.6,
        (total >= 0.3) & (total < 0.6),
        (total > -0.1) & (total < 0.3),
        total <= -0.1,
    ]
    choices = ["No Nap", "Boost 🔥", "Good ✅", "Neutral 🤷‍♂️", "Disrupt ⚠️"]
    return np.select(conditions, choices, default="Neutral 🤷‍♂️")


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------
def fetch_sleep_from_sheets() -> pd.DataFrame:
    """Discover the ``Health Metrics_v*`` sheets in Drive and stack their Sleep tab.

    Lists matching spreadsheets, reads each one's ``Sleep`` tab, concatenates
    the rows, and orders them so that on duplicate dates the highest sheet
    version wins.

    Returns:
        The combined sleep DataFrame across all matching sheets.

    Raises:
        RuntimeError: If no matching sheets are found or none could be read.
    """
    # Import diferido: las funciones de transformación de este módulo deben poder
    # usarse y testearse sin las dependencias de Google ni credenciales.
    from ..gsheets import get_services

    sheets_service, drive_service = get_services()

    query = (
        f"name contains '{HEALTH_METRICS_PREFIX}' "
        "and mimeType='application/vnd.google-apps.spreadsheet' "
        "and trashed=false"
    )
    found = (
        drive_service.files()
        .list(q=query, fields="files(id, name)", orderBy="name")
        .execute()
        .get("files", [])
    )
    if not found:
        raise RuntimeError(
            f"No hay Google Sheets cuyo nombre contenga '{HEALTH_METRICS_PREFIX}'. "
            "Verificá que la service account tenga acceso compartido a esas hojas."
        )

    print(f"Encontradas {len(found)} hoja(s) de Health Metrics:")
    for f in found:
        print(f"  • {f['name']}  (id: {f['id']})")

    frames = []
    for f in found:
        try:
            values = (
                sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=f["id"], range=f"{SLEEP_TAB}!A1:Z")
                .execute()
                .get("values", [])
            )
            if len(values) < 2:
                print(f"  ⚠️  '{f['name']}' tiene la pestaña Sleep vacía — se omite.")
                continue
            frame = pd.DataFrame(values[1:], columns=values[0])
            frame.columns = frame.columns.str.strip()
            frame["_source_sheet"] = f["name"]
            frames.append(frame)
            print(f"  ✅ {len(frame)} filas de '{f['name']}'")
        except Exception as e:
            print(f"  ⚠️  No se pudo leer '{f['name']}': {e}")

    if not frames:
        raise RuntimeError("No se pudo cargar datos de ninguna hoja de Health Metrics.")

    # Ante fechas duplicadas gana la hoja de versión más alta.
    combined = pd.concat(frames, ignore_index=True)
    order = {f["name"]: i for i, f in enumerate(found)}
    combined["_source_order"] = combined["_source_sheet"].map(order)
    combined = combined.sort_values("_source_order", ascending=False)
    return combined.drop(columns=["_source_order", "_source_sheet"])


# ---------------------------------------------------------------------------
# Transformación
# ---------------------------------------------------------------------------
def clean_sleep(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split main sleep from naps and normalize durations.

    Parses dates, drops unused columns, converts duration columns to minutes
    (then main-sleep stages to hours), and separates the ``Main`` rows from
    nap rows, deduplicating each by date.

    Args:
        raw: Raw stacked sleep DataFrame from :func:`fetch_sleep_from_sheets`.

    Returns:
        A ``(main, naps)`` tuple of DataFrames: nightly main sleep and
        per-day naps.
    """
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.rename(columns={"Core": "Light"})

    for col in DURATION_COLUMNS:
        df[col] = df[col].apply(parse_sleep_duration)
    for col in ["REM", "Light", "Deep", "Wake Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    naps = df[df["Main"] == "FALSE"].copy()
    naps = naps.sort_values("Date", ascending=False)
    naps = naps[["Date", "Main", "Start", "End", "Asleep", "Data Source"]]
    naps = naps.drop_duplicates(subset=["Date"])
    naps.columns = [
        "Date",
        "Main",
        "Start_Nap",
        "End_Nap",
        "Asleep_Nap",
        "Data Source Nap",
    ]

    main = df[df["Main"] == "TRUE"].copy()
    main = main.sort_values("Date", ascending=False).drop_duplicates(subset=["Date"])
    for col in ["InBed", "Asleep", "REM", "Light", "Deep"]:
        main[col] = np.round(main[col], 2) / 60

    return main, naps


def load_hrv(path: Path = HRV_XLSX) -> pd.DataFrame:
    """Load the HRV Excel export and normalize its columns.

    Strips the ``ms`` suffix from HRV columns, coerces them to numeric, and
    parses the date column.

    Args:
        path: Path to the HRV ``.xlsx`` file. Defaults to ``HRV_XLSX``.

    Returns:
        The cleaned HRV DataFrame.
    """
    hrv = pd.read_excel(path)
    for col in ["Overnight HRV", "7d Avg"]:
        hrv[col] = hrv[col].astype(str).str.replace("ms", "", regex=False).str.strip()
        hrv[col] = pd.to_numeric(hrv[col], errors="coerce")
    hrv["Date"] = pd.to_datetime(hrv["Date"], format="%Y-%m-%d", errors="coerce")
    return hrv


def load_garmin_sleep(path: Path = GARMIN_SLEEP_XLSX) -> pd.DataFrame:
    """Load the Garmin sleep Excel export and normalize its date column.

    Renames the leading column to ``Date``, drops duplicate dates, parses the
    date, and sorts most-recent first.

    Args:
        path: Path to the Garmin sleep ``.xlsx`` file. Defaults to
            ``GARMIN_SLEEP_XLSX``.

    Returns:
        The cleaned Garmin sleep DataFrame.
    """
    garmin = pd.read_excel(path)
    garmin = garmin.rename(columns={"Sleep Score 4 Weeks": "Date"})
    garmin = garmin.drop_duplicates(subset=["Date"])
    garmin["Date"] = pd.to_datetime(garmin["Date"], errors="coerce")
    return garmin.sort_values("Date", ascending=False)


def build_recovery(merged: pd.DataFrame, garmin: pd.DataFrame) -> pd.DataFrame:
    """Compute the Sigmoid Recovery Score and its nap adjustment.

    Merges Garmin data, z-scores the recovery components (inverting resting
    heart rate and previous-day stress so higher is worse), averages them
    into a raw score, and maps it through a sigmoid. Then derives nap timing
    and duration scores and a nap-adjusted score.

    Args:
        merged: Sleep DataFrame already joined with HRV.
        garmin: Cleaned Garmin sleep DataFrame from :func:`load_garmin_sleep`.

    Returns:
        The recovery DataFrame with score, nap and delta columns added.
    """
    health = pd.merge(merged, garmin, how="left", on="Date")
    health = health.sort_values("Date", ascending=True)

    # El estrés del día previo es lo que afecta al sueño de esta noche.
    health["Stress_prev_day"] = health["Stress"].shift(1)

    for col in RECOVERY_COMPONENTS:
        health[col] = pd.to_numeric(health[col], errors="coerce")
        health["Z " + col] = (health[col] - health[col].mean()) / health[col].std(ddof=0)

    # Más pulsaciones en reposo y más estrés son peores: se invierte el signo.
    health["Z Resting Heart Rate"] = -health["Z Resting Heart Rate"]
    health["Z Stress_prev_day"] = -health["Z Stress_prev_day"]

    # sum(axis=1) ignoraría los NaN: si falta un componente el score debe ser NaN,
    # no una media silenciosamente sesgada.
    health["RECOVERY_SCORE_RAW"] = (
        health[["Z " + c for c in RECOVERY_COMPONENTS]].sum(axis=1, skipna=False) / 5
    )
    health["Sigmoid Recovery Score"] = 1 / (1 + np.exp(-health["RECOVERY_SCORE_RAW"]))

    end_nap = pd.to_datetime(health["End_Nap"], format="%H:%M", errors="coerce")
    health["End_Nap_Decimal"] = end_nap.dt.hour + end_nap.dt.minute / 60
    start_nap = pd.to_datetime(health["Start_Nap"], format="%H:%M", errors="coerce")
    health["Start_Nap_Decimal"] = start_nap.dt.hour + start_nap.dt.minute / 60

    health["Nap_Classifier"] = health["Start_Nap_Decimal"].apply(classify_nap)
    health["Nap_Duration_Score"] = health["Asleep_Nap"].apply(nap_duration)
    health["Nap Status"] = nap_status(health["Nap_Classifier"], health["Nap_Duration_Score"])
    health["Sigmoid with Nap"] = (
        health["Sigmoid Recovery Score"] + health["Nap_Classifier"] + health["Nap_Duration_Score"]
    ).clip(0, 1)
    health["DELTA_NAP"] = health["Sigmoid with Nap"] - health["Sigmoid Recovery Score"]
    return health


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run(
    sleep_output: Path = CLEAN_SLEEP_CSV, recovery_output: Path = CLEAN_RECOVERY_CSV
) -> tuple[Path, Path]:
    """Run the sleep ingestion step end to end and write the CSV outputs.

    Fetches and cleans sleep, joins naps and HRV, builds the recovery
    dataset, and writes the clean sleep and recovery CSVs.

    Args:
        sleep_output: Destination path for the clean sleep CSV. Defaults to
            ``CLEAN_SLEEP_CSV``.
        recovery_output: Destination path for the clean recovery CSV.
            Defaults to ``CLEAN_RECOVERY_CSV``.

    Returns:
        A ``(sleep_output, recovery_output)`` tuple of the written paths.
    """
    ensure_dirs()

    main, naps = clean_sleep(fetch_sleep_from_sheets())

    sleep_with_naps = pd.merge(main, naps, how="left", on="Date")
    sleep_with_naps = sleep_with_naps.drop_duplicates(subset=["Date"])

    merged = pd.merge(sleep_with_naps, load_hrv(), how="left", on="Date")
    merged = merged.sort_values("Date", ascending=False).drop_duplicates(subset=["Date"])
    for col in ["InBed", "Asleep", "REM", "Light", "Deep"]:
        merged = merged.rename(columns={col: f"{col} hrs"})

    merged.to_csv(sleep_output, index=False)
    print(f"💾 Sueño limpio ({len(merged)} filas) -> {sleep_output}")

    health = build_recovery(merged, load_garmin_sleep())
    health.to_csv(recovery_output, index=False)
    print(f"💾 Recovery limpio ({len(health)} filas) -> {recovery_output}")

    return sleep_output, recovery_output


if __name__ == "__main__":
    run()
