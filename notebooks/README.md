# notebooks/

Notebooks de exploración y prototipado. Lo que se consolida termina en
`src/rehab_strength/` o en la app.

## Por qué los `.ipynb` no se versionan

Los outputs guardados embeben datos de salud personales: tablas con fechas y
métricas, y gráficos renderizados. `.gitignore` bloquea `notebooks/*.ipynb` y el
hook `nbstripout` limpia los outputs de cualquier notebook que llegue a un commit.

Si querés publicar uno concreto, limpialo primero y forzá el add:

```bash
nbstripout notebooks/mi_analisis.ipynb
git add -f notebooks/mi_analisis.ipynb
```

Revisá el diff antes de commitear: `nbstripout` quita los outputs, pero **no** las
rutas absolutas ni los datos escritos en las celdas de código.

## Convención

Importá el paquete en vez de duplicar lógica:

```python
from rehab_strength.config import CLEAN_RECOVERY_CSV
from rehab_strength.ingest.sleep import classify_nap

import pandas as pd
recovery = pd.read_csv(CLEAN_RECOVERY_CSV)
```
