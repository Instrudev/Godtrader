# Changelog — Branch remediation/v1

## Tests pre-existentes con fallo conocido

Ver `KNOWN_ISSUES.md` para detalle completo. 3 tests excluidos de la métrica de remediación:
- `test_measure_streaks_ignores_doji` (bug en streaks, severidad baja)
- `test_classifier_no_model_returns_neutral` (auto-carga oculta de ML, auditar en Tarea 1.5)
- `test_classifier_predict_proba_from_df_no_model` (idem)

Baseline de remediación: **250 passed, 3 known failures**. Cualquier fallo adicional es regresión.

---

## Tarea 0.5 — Baseline auditable del modelo ML (2026-04-30)

Snapshot inmutable del modelo en producción creado en `models/audit_baseline/`.
- `lgbm_model_pre_audit_20260430_052304_UTC.pkl` (SHA256: `f1709cb4...`)
- `platt_calibrator_pre_audit_20260430_052304_UTC.pkl` (SHA256: `5ef1910a...`)
- Verificación: `cd models/audit_baseline/ && shasum -a 256 -c baseline_hashes.txt`

Modelo original creado 2026-04-28 21:42 UTC con `train_model.py` manual. Sin versionado ni métricas registradas.

---

## Tarea 0.4 — MIN_WINRATE a 0.55 (2026-04-30)

- `MIN_WINRATE_HOURLY`: 0.39 → 0.55
- `MIN_WINRATE_WEEKDAY`: 0.39 → 0.55
- Breakeven con payout 80-83% ≈ 54.6%. Umbral 0.55 da margen de seguridad.
- Temporal durante remediación. Tarea 2.4 implementa walk-forward definitivo.

---

## Alcance ampliado de Tarea 1.5 — Auditoría ML (documentado 2026-04-30)

Hallazgo durante Fase 0: `ml_classifier.predict_proba()` auto-carga `models/lgbm_model.pkl` sin verificación de integridad ni logging. Adiciones al alcance original:

1. **Verificación de hash SHA256**: Al cargar el modelo, comparar hash contra baseline conocido. Si no coincide, loguear warning y reportar.
2. **Logging de modelo cargado**: En startup y en cada predicción crítica, loguear path, hash truncado y fecha de modificación del modelo activo.
3. **Evaluar bloqueo de auto-carga**: Considerar reemplazar auto-carga implícita (en `predict_proba`) por carga explícita (solo en startup o por trigger). Documentar decisión con pros/contras.
4. **Verificar consistencia**: Comparar modelo en `models/lgbm_model.pkl` contra el baseline en `models/audit_baseline/` usando hashes.

Estos cambios se implementarán en Tarea 1.5, no antes.

---

## Tarea 0.2 — Halt Operativo (2026-04-30)

### Cambios

**iqservice.py:**
- Agregadas constantes `REMEDIATION_MODE = True`, `FORCE_DEMO_ACCOUNT = True`, `ALLOWED_ACCOUNT_TYPES = {"PRACTICE"}`.
- Nuevo método `get_account_type()`: consulta al broker el tipo de cuenta real (no depende del monkey-patch de `change_balance`).
- Nuevo método privado `_enforce_demo_guard()`: guard fail-closed que verifica tipo de cuenta antes de cada orden. Lanza `RuntimeError` si:
  - La cuenta no es PRACTICE.
  - No se puede determinar el tipo de cuenta (fail-closed).
  - El tipo de cuenta es desconocido (whitelist, no blacklist).
- Guard insertado en `buy_binary()` y `buy_digital()` — los únicos 2 caminos de ejecución real.
- Logger de seguridad dedicado (`logs/security_halts.log`) que registra cada bloqueo con timestamp UTC, asset, dirección, tipo de cuenta, stack trace.
- Banner de startup en `connect()` con: modo (REMEDIATION/PRODUCTION), tipo de cuenta, estado del guard, confirmación de match.

**main.py:**
- Banner de warning en `lifespan()` si `REMEDIATION_MODE` es True.

### Tests agregados

**tests/test_halt_operativo.py** (11 tests):
1. `test_guard_blocks_real_account` — cuenta REAL bloqueada en buy_binary
2. `test_guard_blocks_digital_real_account` — cuenta REAL bloqueada en buy_digital
3. `test_guard_allows_practice_account` — cuenta PRACTICE permitida
4. `test_guard_disabled_allows_real` — guard desactivado permite cuenta REAL
5. `test_remediation_mode_constant` — constante es True en esta branch
6. `test_connect_logs_startup_warning` — banner de seguridad en connect()
7. `test_guard_blocks_when_account_type_unavailable` — excepción en verificación → bloqueo
8. `test_guard_blocks_unknown_account_type` — tipo "REAL_V2" desconocido → bloqueo
9. `test_guard_blocks_case_sensitive` — "practice" minúsculas → normaliza a PRACTICE → pasa
10. `test_guard_blocks_none_account_type` — None → bloqueo
11. `test_force_demo_overrides_remediation_disabled` — FORCE_DEMO independiente de REMEDIATION_MODE

### Archivos no modificados

- `asset_scanner.py`, `trader.py`, `paper_trader.py` — no necesitan cambios (todos pasan por `iqservice.buy_binary`/`buy_digital`).
- `paper_trader.py` — simula sin tocar el broker, no necesita guard.

---

## Cambios pre-plan preservados (2026-04-30)

Branch archivo: `pre-remediation-changes` (no mergeable)

Cambios hechos antes del plan de remediación, preservados para auditoría formal en sus tareas correspondientes.

| Archivo | Cambio | Tarea de re-evaluación |
|---|---|---|
| `indicators.py` | RSI 35/65 → 30/70 en `pre_qualify_classical()` | 2.2 |
| `indicators.py` | BB body reversal: `bb_mid` → `bb_upper` en `detect_bb_body_reversal()` | 2.3 |
| `asset_scanner.py` | RSI 30/70 en `_infer_direction()` y `_score_signal()` | 2.2 |
| `asset_scanner.py` | MIN_PROBABILITY 48 → 55, threads 5→3, dedup assets, reconnect logic | 2.2 / 2.4 |
| `regime_filter.py` | Whitelist forex (ALLOWED_ASSETS), BLOCKED_HOURS/WEEKDAYS vaciados | 2.4 |
| `regime_filter.py` | BLOCKED_ASSETS ampliado (+AUDCAD-OTC, +GBPJPY-OTC) | 2.4 |
| `ml_classifier.py` | Ajuste menor (3 líneas) | 1.5 / 1.7 |
| `tests/test_regime_filter.py` | _safe_datetime → Lunes 15h, asset "X" → "EURUSD-OTC" | 2.4 |
| `tests/test_ml_classifier.py` | Adaptación a cambios de ml_classifier | 1.5 |

Cada cambio será auditado individualmente en su tarea correspondiente.
Pueden ser cherry-picked, re-implementados o descartados según decisión arquitectónica.

---

## Cómo salir del modo remediación

Cuando la remediación esté completa y todas las fases hayan pasado sus criterios de aceptación:

1. En `iqservice.py`, cambiar:
   ```python
   REMEDIATION_MODE: bool = False    # Desactivar modo remediación
   FORCE_DEMO_ACCOUNT: bool = False  # Permitir ejecución en cuenta real
   ```

2. Verificar que el banner de startup muestra `Mode: PRODUCTION` y `Demo Guard: DISABLED`.

3. Ejecutar la suite completa de tests: `python -m pytest -x -q`.

4. Confirmar que el bot opera en cuenta PRACTICE del broker antes de pasar a REAL.

5. Para volver a activar el guard en cualquier momento (emergencia), basta con:
   ```python
   FORCE_DEMO_ACCOUNT: bool = True
   ```
   Esto bloquea inmediatamente todas las órdenes sin necesidad de reiniciar (el guard se evalúa en cada llamada a `buy_binary`/`buy_digital`).
