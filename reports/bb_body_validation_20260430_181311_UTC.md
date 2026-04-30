# BB Body Reversal Validation Report

Generated: 2026-04-30 18:13:11 UTC

## 1. Rendimiento Histórico (trades reales)

**4 trades** ejecutados con BB Body Reversal (todos PUT):

| Asset | Dir | Result |
|---|---|---|
| `GBPUSD-OTC` | PUT | LOSS |
| `XAUUSD-OTC` | PUT | LOSS |
| `RAYDIUMUSD-OTC` | PUT | LOSS |
| `EURUSD-OTC` | PUT | WIN |

**Winrate: 1/4 = 25%** (breakeven ~55%)

## 2. Backtest sobre velas históricas

**Disclaimer:** Backtest sobre 1245 velas en 4 assets.
Cada asset cubre ~5 horas. Resultados indicativos, no concluyentes.

### EURUSD-OTC (345 velas)
- PUT señales: 17 (confirmadas: 7)
- CALL señales: 24 (confirmadas: 10)

### HK33-OTC (300 velas)
- PUT señales: 62 (confirmadas: 30)
- CALL señales: 31 (confirmadas: 15)

## 3. Resumen comparativo

| Versión | Señales | Confirmadas | Tasa |
|---|---|---|---|
| PUT (original) | 79 | 37 | 47% |
| CALL (espejo) | 55 | 25 | 45% |

## 4. Conclusión

PUT generó 79 señales, CALL generó 55.
Ambas versiones generan señales. Comparar tasas de confirmación.

## 5. Decisión

**Deprecación instrumentada** de BB Body Reversal como señal activa.
- Rendimiento histórico: 25% WR (1W/3L), por debajo de breakeven.
- Señales phantom registradas en `strategy_decisions.log`.
- Revisión programada tras 30-60 días de operación post-remediación.
- Criterio de reactivación: phantom signals muestran WR > 55% consistente.