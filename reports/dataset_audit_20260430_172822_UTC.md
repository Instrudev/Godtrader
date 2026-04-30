# Dataset Audit Report

Generated: 2026-04-30 17:28:22 UTC

## 1. Estado Actual del Dataset

**Total trades cerrados (WIN/LOSS):** 114

| Resultado | Count |
|---|---|
| LOSS | 55 |
| PENDING | 5 |
| TIE | 3 |
| WIN | 59 |

| Mode | Count |
|---|---|
| live | 82 |
| paper | 40 |

**Rango temporal:** 2026-04-23 → 2026-04-30

| Asset | Trades | W | L | OTC |
|---|---|---|---|---|
| `EURUSD-OTC` | 42 | 24 | 18 | Yes |
| `USDJPY-OTC` | 15 | 9 | 6 | Yes |
| `USDCHF-OTC` | 9 | 5 | 4 | Yes |
| `EURJPY-OTC` | 9 | 6 | 3 | Yes |
| `GBPUSD-OTC` | 4 | 2 | 2 | Yes |
| `AUDCAD-OTC` | 3 | 0 | 3 | Yes |
| `SEIUSD-OTC` | 2 | 2 | 0 | Yes |
| `META/GOOGLE-OTC` | 2 | 0 | 2 | Yes |
| `JUPUSD-OTC` | 2 | 0 | 2 | Yes |
| `XAUUSD-OTC` | 1 | 0 | 1 | Yes |
| `WLDUSD-OTC` | 1 | 1 | 0 | Yes |
| `USNDAQ100-OTC` | 1 | 0 | 1 | Yes |
| `USDSGD-OTC` | 1 | 1 | 0 | Yes |
| `UKOUSD-OTC` | 1 | 0 | 1 | Yes |
| `SNDK-OTC` | 1 | 1 | 0 | Yes |
| `RAYDIUMUSD-OTC` | 1 | 0 | 1 | Yes |
| `ORDIUSD-OTC` | 1 | 0 | 1 | Yes |
| `MORSTAN-OTC` | 1 | 0 | 1 | Yes |
| `KLARNA-OTC` | 1 | 0 | 1 | Yes |
| `IOTAUSD-OTC` | 1 | 0 | 1 | Yes |

| Fecha | Trades | W | L |
|---|---|---|---|
| 2026-04-23 | 13 | 8 | 5 |
| 2026-04-24 | 15 | 9 | 6 |
| 2026-04-25 | 5 | 1 | 4 |
| 2026-04-27 | 30 | 18 | 12 |
| 2026-04-28 | 17 | 9 | 8 |
| 2026-04-29 | 21 | 11 | 10 |
| 2026-04-30 | 13 | 3 | 10 |

## 2. Análisis de Contaminación

**Trades no-OTC:** 2

| ID | Timestamp | Asset | Dir | Result | Mode |
|---|---|---|---|---|---|
| 421 | 2026-04-29T17:16:05 | `EURJPY-op` | PUT | WIN | live |
| 422 | 2026-04-29T17:20:03 | `BXY` | CALL | LOSS | live |

**Trades pre-Tarea 1.0 (sin filtros completos):** 114 (todos)
**Trades post-Tarea 1.0 (con filtros):** 0 (branch nunca desplegada)

## 3. Análisis de Features

| Categoría | Features | Trades con datos | % |
|---|---|---|---|
| Básicas (rsi, bb_*) | 6 | 112 | 98% |
| Streak/ML (streak_*, ret_*, half_life) | 8 | 36 | 32% |
| PRNG (prng_*) | 9 | 53 | 46% |
| **COMPLETAS (todas)** | **23** | **0** | **0%** |

| Pipeline | Trades | Con streak | Con PRNG |
|---|---|---|---|
| scanner | 78 | 0 | 55 |
| trader | 36 | 36 | 0 |

## 4. Análisis del Pipeline `_row_to_features`

**Anti-patrón identificado:** `_row_to_features()` en `train_model.py` rellena NULL con valores default sensibles (streak_length=0, streak_pct=50, ret_*=0, half_life=100, prng_*=0).

**Consecuencia:** 78 trades del scanner tienen valores defaults idénticos en 8 features de racha/retorno y 9 features PRNG. El modelo entrenó con 68% del dataset sin varianza informativa real en esas features.

**Implicación directa:** Explica por qué 21/27 features tienen importancia 0 en el modelo actual (Tarea 1.5). El modelo no puede aprender de features que son constantes en 2/3 del dataset.

## 5. Conclusión

**Dataset NO viable para reentrenamiento inmediato.** Causa raíz: combinación de dataset pequeño (114 trades), features incompletas (0 trades con features completas), y anti-patrón de defaults que elimina varianza informativa.

## 6. Especificación de Tarea 3.1 (Reentrenamiento)

### Criterios de activación (mínimos)
- 200 trades con features COMPLETAS (no rellenadas con defaults).
- Bot operando con `remediation/v1` desplegado (filtros + features pobladas).
- Distribución balanceada: al menos 30% WIN y 30% LOSS.
- Datos de al menos 30 días de operación continua.

### Pipeline a revisar
- `_row_to_features` debe NO rellenar NULL con defaults.
- Opciones: dropear filas con NULL, imputación por mediana, o flag explícito.
- `asset_scanner` debe poblar TODAS las features (streak + PRNG) en cada trade.

### Validación obligatoria
- Walk-forward temporal (no aleatoria).
- `metrics.json` guardado en cada entrenamiento.
- Feature importances logueadas.
- AUC >= 0.62 en test set.
- `direction` debe estar entre top 5 features por importancia.

### Arquitectura alternativa a considerar
- 2 modelos separados (CALL y PUT) si `direction` sigue con importancia 0.
- Eliminar features PRNG si siguen mostrando importancia ~0 con datos completos.

## 7. Estimación Temporal
- Configuración Tarea 1.6: max 5 trades/día.
- Realista: 1-3 trades/día (score >= 0.75 es selectivo).
- Para 200 trades: 70-200 días (2-6 meses).
- Para 500 trades: 170-500 días (6-18 meses).
- **Recomendación:** revisar criterios cada 30 días de operación.

## 8. Issues Conocidos para Tarea 3.1

| # | Issue | Impacto | Archivo |
|---|---|---|---|
| 1 | `_row_to_features` rellena NULL con defaults | 21/27 features con importancia 0 | `train_model.py` |
| 2 | Pipeline scanner no popula features streak | 78 trades sin streak_length/pct/ret_* | `asset_scanner.py` |
| 3 | Pipeline trader no popula features PRNG | 36 trades sin prng_* | `trader.py` (deprecated) |
| 4 | `train_model.py` no guarda `metrics.json` | Sin historial de AUC/Brier | `train_model.py` |

## 9. Trades Excluidos de Futuro Reentrenamiento

Trades no-OTC a excluir permanentemente:

| ID | Asset | Razón |
|---|---|---|
| 421 | `EURJPY-op` | Mercado real (no OTC) |
| 422 | `BXY` | Bloomberg Dollar Index (no OTC, no forex) |