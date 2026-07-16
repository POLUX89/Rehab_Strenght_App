# models/

Modelos entrenados que exporta la app. **No se versionan** (`.gitignore` bloquea
`*.pkl`, `*.joblib`, `*.h5`, `*.onnx`): son artefactos regenerables y un pickle
entrenado puede filtrar los datos con los que se ajustó.

## Convención de nombres

```
{objetivo}_{algoritmo}_v{version}.pkl      # sleep_quality_logit_v1.pkl
```

## Exportar un modelo desde la app

```python
import joblib
from rehab_strength.config import MODELS_DIR, ensure_dirs

ensure_dirs()
joblib.dump(model, MODELS_DIR / "sleep_quality_logit_v1.pkl")
```

Un pickle solo se puede cargar con la misma versión de scikit-learn con que se
guardó. Si vas a conservar un modelo, anotá en el `CHANGELOG.md` con qué datos y
qué versión de la librería se entrenó.
