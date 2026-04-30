# ML Model Audit Baseline

Snapshot inmutable del modelo ML en producción antes de la auditoría formal (Tarea 1.5).

## Archivos

| Archivo | SHA256 |
|---|---|
| `lgbm_model_pre_audit_20260430_052304_UTC.pkl` | `f1709cb4359a9157b97fc1b6599cf5b51d6f127f124afb69f79081206f7ebfe8` |
| `platt_calibrator_pre_audit_20260430_052304_UTC.pkl` | `5ef1910a1b7fe839e5d2255ffb88d4fafb328d3eef59d8e33b308b29ea8b08aa` |

## Estado conocido

- **Fecha de creación original**: 2026-04-28 21:42:18 UTC
- **Método de creación**: Manual con `python train_model.py`
- **No creado por** `retrain_scheduler` (directorio `models/versions/` vacío)
- **Sin métricas de creación registradas**: No hay log de AUC, Brier, ni tamaño del dataset usado
- **Sin versionado original**: No existía snapshot previo
- **Modelo**: LightGBM binary classifier + Platt scaling (LogisticRegression)
- **Features**: 30 (ver `ml_classifier.FEATURE_COLS`)

## Propósito

Snapshot inmutable para comparación en Tarea 1.5 (Auditoría ML).
No modificar ni eliminar estos archivos.

## Verificación de integridad

```bash
cd models/audit_baseline/
shasum -a 256 -c baseline_hashes.txt
```
