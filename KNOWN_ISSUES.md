# Known Issues

Tests pre-existentes con fallo conocido. No se corrigen durante la remediación.
La métrica de "tests passing" para tareas de remediación excluye estos 3 tests.
Si aparece un cuarto fallo, se trata como regresión y es bloqueante.

| Test | Archivo | Causa raíz | Severidad | Tarea futura |
|---|---|---|---|---|
| `test_measure_streaks_ignores_doji` | `tests/test_indicators_otc.py:241` | Bug en lógica de `_measure_all_streaks`: dojis rompen la racha en vez de ignorarlos | Baja | Issue separado post-remediación |
| `test_classifier_no_model_returns_neutral` | `tests/test_ml_classifier.py:214` | Auto-carga oculta: `predict_proba()` llama `self.load()` implícitamente, detecta `models/lgbm_model.pkl` en disco y devuelve predicciones reales en vez de 0.5 | Media | Auditar en Tarea 1.5 |
| `test_classifier_predict_proba_from_df_no_model` | `tests/test_ml_classifier.py:240` | Misma causa: auto-carga oculta del modelo desde disco | Media | Auditar en Tarea 1.5 |

## Notas adicionales para Tarea 1.5

- El modelo actual (`models/lgbm_model.pkl`) fue creado el 2026-04-28 21:42 con `train_model.py`, no por `retrain_scheduler`.
- El directorio `models/versions/` está vacío — no hay historial de versiones.
- No hay verificación de hash ni integridad del modelo al cargar.
- La auto-carga implícita en `predict_proba()` puede causar carga de modelos obsoletos o corruptos sin aviso.
