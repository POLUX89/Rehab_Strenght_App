"""Corre el pipeline completo de ingesta.

python -m rehab_strength.ingest.run_all
"""

from __future__ import annotations

import sys
from datetime import date

from . import sleep, strong


def main() -> int:
    """Run the full ingestion pipeline (sleep, then workouts).

    Executes each step in order, catching and reporting failures per step so
    one broken source does not abort the rest.

    Returns:
        Process exit code: ``0`` if every step succeeded, ``1`` if any
        step raised.
    """
    print(f"\n🚀 Ingesta completa — {date.today()}")

    steps = [
        ("📥 Sueño desde Google Sheets...", sleep.run),
        ("📥 Workouts desde el CSV de Strong...", strong.run),
    ]

    failed = []
    for label, step in steps:
        print(f"\n{label}")
        try:
            step()
        except Exception as e:
            print(f"❌ Falló: {e}", file=sys.stderr)
            failed.append(step.__module__)

    if failed:
        print(f"\n❌ Pasos fallidos: {', '.join(failed)}", file=sys.stderr)
        return 1

    print("\n✅ Datos importados y procesados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
