"""Autenticación con Google contra Sheets y Drive.

Las credenciales nunca viven en el código ni en el repo. Se resuelven en este orden:

1. ``GCP_SERVICE_ACCOUNT_JSON``      — el JSON completo como string (CI, contenedores)
2. ``GOOGLE_APPLICATION_CREDENTIALS`` — ruta al archivo JSON (desarrollo local)
3. ``st.secrets["gcp_service_account"]`` — tabla TOML (Streamlit Cloud)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from .config import SHEETS_SCOPES


class CredentialsNotFoundError(RuntimeError):
    """No se encontró ninguna credencial utilizable."""


def _from_streamlit_secrets() -> dict[str, Any] | None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return None
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        # Fuera de un runtime de Streamlit, acceder a st.secrets lanza excepción.
        return None
    return None


def _service_account_info() -> dict[str, Any]:
    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if raw:
        return json.loads(raw)

    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise CredentialsNotFoundError(
                f"GOOGLE_APPLICATION_CREDENTIALS apunta a {p}, que no existe."
            )
        return json.loads(p.read_text())

    info = _from_streamlit_secrets()
    if info:
        return info

    raise CredentialsNotFoundError(
        "No hay credenciales de Google. Definí GCP_SERVICE_ACCOUNT_JSON o "
        "GOOGLE_APPLICATION_CREDENTIALS, o agregá [gcp_service_account] en "
        ".streamlit/secrets.toml. Ver .env.example."
    )


def get_credentials() -> Credentials:
    return Credentials.from_service_account_info(_service_account_info(), scopes=SHEETS_SCOPES)


def get_services() -> tuple[Any, Any]:
    """Devuelve los clientes ``(sheets, drive)`` ya autenticados."""
    creds = get_credentials()
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    return sheets, drive
