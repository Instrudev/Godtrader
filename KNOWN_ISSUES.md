# Known Issues

Tests pre-existentes con fallo conocido. No se corrigen durante la remediación.
La métrica de "tests passing" para tareas de remediación excluye este test.
Si aparece un segundo fallo, se trata como regresión y es bloqueante.

| Test | Archivo | Causa raíz | Severidad | Tarea futura |
|---|---|---|---|---|
| `test_measure_streaks_ignores_doji` | `tests/test_indicators_otc.py:241` | Bug en lógica de `_measure_all_streaks`: dojis rompen la racha en vez de ignorarlos | Baja | Issue separado post-remediación |

## Resueltos en Tarea 1.5

Los siguientes tests que fallaban por auto-carga oculta de ML ahora pasan gracias al bloqueo de auto-carga en REMEDIATION_MODE:

- `test_classifier_no_model_returns_neutral` — ahora pasa (auto-carga bloqueada)
- `test_classifier_predict_proba_from_df_no_model` — ahora pasa (auto-carga bloqueada)
