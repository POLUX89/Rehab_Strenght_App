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
    """Read service-account credentials from ``st.secrets`` if available.

    Returns:
        The ``gcp_service_account`` secrets table as a dict, or None when
        Streamlit is not installed, there is no active Streamlit runtime, or
        the key is absent.
    """
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
    """Resolve service-account credentials from the first available source.

    Tries, in order: the ``GCP_SERVICE_ACCOUNT_JSON`` inline JSON, the file
    at ``GOOGLE_APPLICATION_CREDENTIALS``, then Streamlit secrets.

    Returns:
        The parsed service-account info as a dict.

    Raises:
        CredentialsNotFoundError: If no source yields usable credentials, or
            ``GOOGLE_APPLICATION_CREDENTIALS`` points to a missing file.
    """
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
    """Build Google service-account credentials scoped to Sheets and Drive.

    Returns:
        A ``Credentials`` object built from the resolved service-account
        info and the read-only Sheets/Drive scopes.

    Raises:
        CredentialsNotFoundError: If no usable credentials can be resolved.
    """
    return Credentials.from_service_account_info(_service_account_info(), scopes=SHEETS_SCOPES)


def get_services() -> tuple[Any, Any]:
    """Build authenticated Google Sheets and Drive API clients.

    Returns:
        A ``(sheets, drive)`` tuple of API client objects with discovery
        caching disabled.
    """
    creds = get_credentials()
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    return sheets, drive
