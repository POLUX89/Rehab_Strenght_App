"""Ingesta del CSV exportado por la app Strong -> workouts limpios."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import CLEAN_WORKOUTS_CSV, STRONG_CSV, ensure_dirs

COLUMNS = [
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
]

# Sets que no cuentan como trabajo efectivo: Warmup, Rest, Failure, Rest Timer.
NON_WORKING_SETS = {"W", "R", "F", "Rest Timer"}

# Por encima de esto es error de captura, no un levantamiento real.
MAX_PLAUSIBLE_WEIGHT_LBS = 900


def parse_duration_to_minutes(value) -> float | None:
    """Convierte '1h 5m', '45m', 'HH:MM:SS' o 'MM:SS' a minutos totales."""
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    if not s:
        return pd.NA

    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h * 60 + m + sec / 60
        if len(parts) == 2:
            m, sec = parts
            return m + sec / 60

    hours = minutes = 0
    if h_match := re.search(r"(\d+)\s*h", s):
        hours = int(h_match.group(1))
    if m_match := re.search(r"(\d+)\s*m", s):
        minutes = int(m_match.group(1))
    return hours * 60 + minutes


def clean_workouts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["SET_ORDER"] = df["SET_ORDER"].astype(str)
    df = df.set_index("DATE").sort_index()

    df["NOTES"] = df["NOTES"].fillna("")
    df = df.drop(columns=["WORKOUT_NOTES", "DISTANCE", "SECONDS"], errors="ignore")
    df = df.dropna(subset=["EXERCISE_NAME"])
    df = df[~df["SET_ORDER"].isin(NON_WORKING_SETS)].copy()

    for col in ["WEIGHT_LBS", "REPS", "RPE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Máscara, no drop(index=...): DATE es el índice y todos los sets de un mismo
    # workout comparten timestamp, así que dropear por índice borraría el workout
    # entero en vez del set mal capturado.
    df = df[~(df["WEIGHT_LBS"] > MAX_PLAUSIBLE_WEIGHT_LBS)].copy()

    df["VOLUME"] = df["WEIGHT_LBS"] * df["REPS"]
    df["RECORDED_RPE"] = df["RPE"].notna()
    # Útil para ejercicios de peso corporal (pushups, chinups).
    df["REPS_ONLY"] = df["REPS"]
    df["DURATION_MIN"] = df["DURATION_MIN"].apply(parse_duration_to_minutes)

    return df.reset_index()


def run(input_path: Path = STRONG_CSV, output_path: Path = CLEAN_WORKOUTS_CSV) -> Path:
    ensure_dirs()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el export de Strong en {input_path}. "
            "Exportá el CSV desde la app y dejalo ahí."
        )

    df = pd.read_csv(input_path, names=COLUMNS, header=0)
    clean = clean_workouts(df)
    clean.to_csv(output_path, index=False)
    print(f"✅ Workouts limpios ({len(clean)} filas) -> {output_path}")
    return output_path


if __name__ == "__main__":
    run()
