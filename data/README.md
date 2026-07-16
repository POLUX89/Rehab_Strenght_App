# data/

**Ningún archivo de datos de esta carpeta se versiona.** Son datos de salud
personales; `.gitignore` los bloquea y solo se conserva la estructura.

| Carpeta      | Contenido                                     | Origen                        |
| ------------ | --------------------------------------------- | ----------------------------- |
| `raw/`       | Exports sin tocar, tal como salen de la fuente | Strong, Garmin, HealthFit     |
| `processed/` | Datasets limpios que consume la app            | Genera `rehab_strength.ingest` |
| `external/`  | Datos de terceros o de referencia              | Manual                        |

## Archivos esperados en `raw/`

- `strong.csv` — export de la app Strong
- `HRV_status.xlsx` — export de HRV de Garmin
- `Sleep_garmin.xlsx` — export de sueño de Garmin

## Salidas en `processed/`

- `clean_strong_workouts.csv`
- `clean_sleep_data.csv`
- `clean_recovery_data.csv`

Regenerá todo con `make ingest`.
