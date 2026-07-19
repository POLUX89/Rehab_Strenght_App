"""Rutas y configuración del proyecto.

Todo se resuelve relativo a la raíz del repo, nunca con rutas absolutas.
Cualquier valor se puede sobrescribir con variables de entorno (ver .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

# src/rehab_strength/config.py -> raíz del repo
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    # dotenv solo viene con el extra [ingest]. Sin él las variables de entorno
    # del shell siguen funcionando; solo no se lee el archivo .env.
    pass
else:
    # override=False: lo que ya esté exportado en el shell gana sobre el .env.
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _dir_from_env(var: str, default: Path) -> Path:
    """Resolve a directory path from an environment variable.

    Args:
        var: Environment variable name to read.
        default: Path used when the variable is unset.

    Returns:
        The resolved absolute path, with ``~`` expanded.
    """
    return Path(os.getenv(var, default)).expanduser().resolve()


DATA_DIR = _dir_from_env("REHAB_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = _dir_from_env("REHAB_MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = _dir_from_env("REHAB_REPORTS_DIR", PROJECT_ROOT / "reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Entradas
STRONG_CSV = RAW_DIR / "strong.csv"
HRV_XLSX = RAW_DIR / "HRV_status.xlsx"
GARMIN_SLEEP_XLSX = RAW_DIR / "Sleep_garmin.xlsx"

# Salidas del pipeline (son las que la app consume)
CLEAN_WORKOUTS_CSV = PROCESSED_DIR / "clean_strong_workouts.csv"
CLEAN_SLEEP_CSV = PROCESSED_DIR / "clean_sleep_data.csv"
CLEAN_RECOVERY_CSV = PROCESSED_DIR / "clean_recovery_data.csv"

# Google Sheets
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
SLEEP_TAB = os.getenv("REHAB_SLEEP_TAB", "Sleep")
# Coincide con v2, v5 y versiones futuras; excluye la hoja sin versionar.
HEALTH_METRICS_PREFIX = os.getenv("REHAB_HEALTH_METRICS_PREFIX", "Health Metrics_v")


def ensure_dirs() -> None:
    """Create the pipeline's output directories if they do not exist.

    Creates the raw, processed, external, models and figures directories
    (parents included), doing nothing for those already present.
    """
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
