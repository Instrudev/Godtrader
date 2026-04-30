# Changelog — Branch remediation/v1

## Tests pre-existentes con fallo conocido

Ver `KNOWN_ISSUES.md` para detalle completo. 3 tests excluidos de la métrica de remediación:
- `test_measure_streaks_ignores_doji` (bug en streaks, severidad baja)
- `test_classifier_no_model_returns_neutral` (auto-carga oculta de ML, auditar en Tarea 1.5)
- `test_classifier_predict_proba_from_df_no_model` (idem)

Baseline de remediación: **250 passed, 3 known failures**. Cualquier fallo adicional es regresión.

---

## Tarea 1.6 — ML disabled fallback mode (2026-04-30)

### Activación
Consecuencia directa de hallazgos de Tarea 1.5: modelo ML no añade valor predictivo real (78% features importancia=0, `direction` importancia=0, calibrador comprime a ~0.5).

### Modo de operación sin ML
| Parámetro | ML Enabled | ML Disabled |
|---|---|---|
| Gate de entrada | ML proba ≥ 55% | Strategy score ≥ 0.75 |
| Pérdidas/activo | 3 | **2** |
| Pérdidas globales | 6 | **4** |
| Trades/día | 15 | **5** |
| `consecutive_loss` | 3 | 3 (sin cambio) |

### Constantes en `iqservice.py` (5 nuevas)
- `ML_DISABLED_MODE = True`
- `ML_DISABLED_MIN_SCORE = 0.75`
- `ML_DISABLED_MAX_ASSET_LOSSES = 2`
- `ML_DISABLED_MAX_DAILY_LOSSES = 4`
- `ML_DISABLED_MAX_DAILY_TRADES = 5`

### Cambios en `asset_scanner.py`
- Imports de constantes ML_DISABLED al inicio del módulo (Condición 1).
- Límites condicionales pasados a `check_all_filters` según modo.
- Bloque de validación: si `ML_DISABLED`, verifica `candidate["score"]` vs `ML_DISABLED_MIN_SCORE`. Si no, requiere ML cargado obligatoriamente.
- Variables `proba`/`pr_pct` inicializadas antes del condicional (fix UnboundLocalError).

### Cambios en `main.py`
- Banner de startup muestra `ML_DISABLED=True/False`.
- Si True: loguea "OPERATING WITHOUT ML — REDUCED RISK MODE" con todos los límites.

### Test de Tarea 1.0 actualizado
`test_scanner_filter_passes_allows_ml` patchea `ML_DISABLED_MODE=False` para probar el flujo ML.

### Reversión
Cambiar `ML_DISABLED_MODE = False` en `iqservice.py` tras reentrenamiento exitoso (Tarea 3.1, AUC ≥ 0.62).

### Tests añadidos (8)
1. `test_ml_disabled_blocks_below_min_score` — score 0.65 < 0.75 → no opera
2. `test_ml_disabled_allows_above_min_score` — score 0.80 ≥ 0.75 → opera
3. `test_ml_required_when_disabled_mode_false` — ML no cargado → no opera
4. `test_ml_disabled_uses_reduced_asset_limit` — max_asset_losses=2
5. `test_ml_disabled_uses_reduced_global_limit` — max_daily_losses=4, auto_shutdown
6. `test_ml_disabled_uses_reduced_daily_trades` — max_trades=5
7. `test_ml_normal_uses_standard_limits` — ML_DISABLED=False → 3/6/15
8. `test_startup_banner_shows_ml_disabled` — banner con ML_DISABLED + limits

---

## Tarea 1.5 — Auditoría ML + carga explícita con verificación SHA256 (2026-04-30)

### Hallazgos estructurales del modelo (FASE A)
1. **21/27 features (78%) con importancia 0** — modelo no aprendió de la mayoría de inputs.
2. **`direction` con importancia 0** — no distingue CALL de PUT. Predicciones son efectivamente iguales.
3. **Top 3 features concentran 90% del peso** — `bb_width_pct` (38%), `rsi` (30%), `bb_pct_b` (22%). Clasificador BB/RSI disfrazado de modelo de 27 features.
4. **Calibrador Platt casi neutro** (coef=0.102) — no corrige significativamente.
5. **9 features PRNG inútiles** — solo 1 de 9 con importancia marginal.
6. **Dataset insuficiente** — 114 trades, 32% con features ML pobladas. Métricas cuantitativas no serían concluyentes.

### Conclusión: MODELO NO AÑADE VALOR PREDICTIVO REAL
Decisión: **Activar Tarea 1.6** (política fallback sin ML).

### Cambios en `ml_classifier.py`
- Nuevo método `load_with_verification(path, expected_hash)` — verifica SHA256 antes de cargar. Log a `security_halts.log` si hash no coincide.
- Logging completo en `load()`: path, hash (12 chars), fecha de modificación, features activas.
- Auto-carga implícita bloqueada en `REMEDIATION_MODE` (resuelve 2 known issues de test).
- `get_model_info()` para snapshot de auditoría.
- Logging checkpoint cada 100 predicciones.

### Reporte generado
`reports/ml_audit_20260430_145654_UTC.md` — análisis estructural completo con disclaimer, feature importances, calibrador, conclusión y recomendaciones.

### Requisitos documentados para Tarea 3.1 (reentrenamiento futuro)
- Mínimo 500 trades con filtros activos (post-Tarea 1.0).
- Excluir trades pre-Tarea 1.0 y no-OTC.
- `train_model.py` debe guardar `metrics.json`.
- `direction` debe estar entre top 5 features.
- Considerar 2 modelos separados (CALL/PUT).
- Eliminar features PRNG si siguen mostrando importancia ~0.

### Known issues resueltos
- `test_classifier_no_model_returns_neutral` — ahora pasa (auto-carga bloqueada en REMEDIATION_MODE).
- `test_classifier_predict_proba_from_df_no_model` — idem.
- Known failures: 3 → 1.

### Tests añadidos (7)
1. `test_load_with_verification_valid_hash`
2. `test_load_with_verification_invalid_hash`
3. `test_auto_load_blocked_in_remediation_mode`
4. `test_load_logs_hash_and_path`
5. `test_audit_script_loads_model_and_validates_hash`
6. `test_audit_script_handles_insufficient_data`
7. `test_load_without_expected_hash`

---

## Tarea 1.4 — Refactorizar lógica de decisión de dirección (2026-04-30)

### Problema resuelto
`_infer_direction()` era cascada con return temprano: la primera estrategia que matcheaba decidía dirección, ignorando señales contradictorias y sesgando hacia BB 2-Candle. `_score_signal()` evaluaba las mismas 4 estrategias por separado, duplicando cómputo.

### Cambios en `asset_scanner.py`
- **Eliminados**: `_infer_direction()` y `_score_signal()` (consolidados).
- **Nuevo**: `_evaluate_strategies(df, asset)` — evalúa las 4 estrategias en paralelo, retorna `(direction, score, signals, resolution)`.
- **Nuevo**: `StrategySignal` frozen dataclass: `(active, direction, score, strategy_name)`.
- **Arbitraje de conflictos**:
  - Todas mismas dirección → `single_match` o `multiple_match` (mayor score gana).
  - Direcciones opuestas → `conflict_cancel` (no opera). Mayoría no gana.
  - Sin señales → `no_signal`.
- **~2x speedup**: cada detector se llama 1 vez en vez de 2 por activo.
- **Telemetría por instancia**: `self._strategy_telemetry` con contadores diarios, reset a 00:00 UTC, `get_strategy_telemetry()`.
- **Log JSONL**: `logs/strategy_decisions.log` — bitácora exhaustiva de todas las evaluaciones.
- **Callsites actualizados**: `_scan_one` y `_confirm_hot` usan `_evaluate_strategies`. `_try_execute` actualiza contadores desde `candidate["resolution"]`.

### Fórmulas de scoring preservadas
Las 4 fórmulas matemáticas se copiaron literalmente de `_score_signal`. Cero cambios a la lógica de scoring.
- BB 2-Candle: `0.70 + min(rsi_extremity, 1.0) * 0.25` (rango 0.70–0.95)
- BB Body: `0.50 + ext*0.28 + rsi_f*0.15 + vol_f` (rango 0.50–0.78)
- Streak: `pct_f*0.75 + len_f*0.25` (rango 0.56–0.75)
- RSI Clásico: `0.40 + rsi_f*0.30 + bb_f*0.30` (rango 0.40–0.70)

### Tests añadidos (10) — `tests/test_scanner_filters.py`
1. `test_no_strategies_active_returns_none`
2. `test_single_strategy_returns_its_direction`
3. `test_multiple_same_direction_returns_highest_score`
4. `test_contradictory_directions_returns_none`
5. `test_3_call_1_put_conflict_cancel` (mayoría no gana)
6. `test_score_from_winning_strategy`
7. `test_telemetry_single_match`
8. `test_telemetry_conflict_cancel`
9. `test_strategy_signal_frozen`
10. `test_backwards_compatible_signature`

### Notas
- RSI usa 35/65 (main HEAD). Tarea 2.2 los cambiará.
- BB Body usa `bb_mid` (main HEAD). Tarea 2.3 lo evaluará.

---

## Tarea 1.3 — Política de TIE en rachas consecutivas (2026-04-30)

### Política definida
`TIE_BREAKS_LOSS_STREAK = True` (default). Un TIE rompe la racha de pérdidas consecutivas.

**Justificación:** Un TIE es un resultado neutral — el patrón perdedor no continuó. Ignorar el TIE (comportamiento legacy) infla artificialmente la racha y puede disparar halt prematuro. Ejemplo: LOSS, LOSS, TIE, LOSS, LOSS antes contaba como 4 consecutivas, ahora cuenta como 2.

**Configurable:** Setear `TIE_BREAKS_LOSS_STREAK = False` en `regime_filter.py` para restaurar comportamiento legacy si se requiere.

### Cambios en `regime_filter.py`
- Nueva constante `TIE_BREAKS_LOSS_STREAK = True`.
- `consecutive_loss_filter`: nuevo parámetro `tie_breaks`. Cuando `True`, TIE entra en la lista evaluada y rompe la racha con `break`. Logging debug cuando TIE rompe.

### Auditoría de otros filtros
| Filtro | Cómo maneja TIE | ¿Cambio necesario? |
|---|---|---|
| `consecutive_loss_filter` | Era invisible → ahora rompe racha | **Sí (esta tarea)** |
| `daily_loss_filter` | Solo cuenta `result == "LOSS"` | No |
| `per_asset_loss_filter` | Solo cuenta `result == "LOSS"` | No |
| `max_trades_filter` | Cuenta `result != "PENDING"` → TIE cuenta como trade | No |

Cada filtro tiene la semántica correcta para su propósito: "TIE no es pérdida" (daily/per-asset), "TIE es trade ejecutado" (max_trades), "TIE rompe patrón perdedor" (consecutive).

### Test de Tarea 1.1 actualizado
`test_consecutive_loss_filter_unchanged_by_utc_migration` actualizado de "TIE invisible" a "TIE rompe racha". El test original preveía este cambio con el comentario: "política actual, cambia en Tarea 1.3".

### Tests añadidos (6)
1. `test_tie_breaks_streak_when_flag_true` — L,L,TIE,L,L → racha=2
2. `test_tie_invisible_when_flag_false` — L,L,TIE,L,L → racha=4
3. `test_default_tie_policy_is_true` — constante es True
4. `test_consecutive_blocks_with_tie_breaks_at_3` — W,L,L,L → racha=3 → bloquea
5. `test_pure_loss_streak_unaffected_by_flag` — L,L,L → ambos flags → bloquea
6. `test_multiple_tie_intercalated` — L,TIE,L,TIE,L → racha=1

---

## Tarea 1.2 — Stop Loss dual por activo + global (2026-04-30)

### Cambios en `regime_filter.py`
- Nueva constante `MAX_ASSET_DAILY_LOSSES = 3` (pérdidas máximas por activo en sesión UTC).
- `MAX_DAILY_LOSSES` subido de 3 → 6 (halt global).
- Nuevo filtro `per_asset_loss_filter(trade_log, asset, max_losses)`: bloquea un activo individual sin detener el bot. Usa `_today_utc()` para reset a 00:00 UTC.
- Insertado en `check_all_filters` como filtro #4 (después de `blocked_asset`, antes de `daily_loss`).
- Nuevo parámetro `max_asset_losses` en firma de `check_all_filters`.

### Cambios en `asset_scanner.py`
- Nuevo método `_reconstruct_trade_log()`: al startup, query a la BD para recuperar trades del día UTC actual. Mitiga vulnerabilidad post-restart donde `trade_log = []` permitiría violar los stop loss.
- Logging de seguridad en `logs/security_halts.log` si la reconstrucción falla (fail-open con auditoría).
- `start()`: `self.trade_log = self._reconstruct_trade_log()` en vez de `= []`.
- Loop de candidatos: `if not self.running: break` — previene ejecución de candidatos adicionales tras auto_shutdown a mitad de ciclo.

### Decisiones arquitectónicas
- **Estado en memoria + reconstrucción BD**: filtros stateless consultan `trade_log` cada vez. Al startup se reconstruye desde BD. Sin persistencia separada de `asset_halted`.
- **Reset diario**: Solo al startup (OPCIÓN 1). Los filtros auto-filtran por fecha UTC.
- **Coexistencia**: `per_asset_loss_filter` (3/activo, bloquea activo), `daily_loss_filter` (6 globales, halt total), `consecutive_loss_filter` (3 racha, halt total) — 3 filtros independientes sin conflicto.

### Observaciones para tareas futuras
- Si `MAX_TRADES_PER_DAY` se sube >30, revisar el cap de 50 en `trade_log` (línea 660).
- Política TIE en racha consecutiva (Tarea 1.3) afectará trades reconstruidos — pero la política se aplica desde el filtro, así que es uniforme.

### Tests añadidos (8)
1. `test_per_asset_3_losses_blocks_that_asset` — 3 LOSS → bloquea asset
2. `test_per_asset_block_allows_other_assets` — 3 LOSS en EURUSD → USDJPY OK
3. `test_global_6_losses_triggers_shutdown` — 6 LOSS distribuidas → halt global
4. `test_global_halt_not_triggered_below_6` — 5 LOSS → no halt
5. `test_per_asset_reset_at_utc_midnight` — LOSS de ayer no cuentan
6. `test_per_asset_and_global_coexist` — per-asset y global funcionan juntos
7. `test_scanner_stops_processing_after_halt` — halt a mitad de ciclo → stop
8. `test_scanner_reconstructs_trade_log_from_db` — startup reconstruye contadores

---

## Tarea 1.0 — Integrar check_all_filters en asset_scanner (2026-04-30)

### Hallazgo crítico (Escenario B)
Durante la verificación de Tarea 1.2, se descubrió que `asset_scanner._try_execute()` **nunca llamaba `check_all_filters()`**. Solo usaba `loss_pattern_filter` (1 de 13 filtros). Los 78 trades ejecutados por el scanner operaron sin:
- `daily_loss_filter` (stop loss diario)
- `consecutive_loss_filter` (racha de pérdidas)
- `blocked_hours_filter` / `blocked_weekday_filter`
- `payout_filter` / `volatility_filter`
- `hour_profile_filter` / `weekday_profile_filter`
- `max_trades_filter` / `min_streak_filter` / `drift_filter`

Esto explica parcialmente el winrate de ~50% observado en los trades del scanner.

**Causa raíz:** El reporte de Tarea 0.1 fue incorrecto. La auditoría leyó el working directory con cambios pre-plan que integraban `check_all_filters`. Al revertir esos cambios, la integración desapareció. El código de `main` HEAD nunca la tuvo.

### Cambios en `asset_scanner.py`
- Import `check_all_filters` desde `regime_filter`.
- Eliminar llamada individual a `loss_pattern_filter` (ahora absorbida como filtro #13 dentro de `check_all_filters`).
- Mover cálculo de `payout` antes de filtros (requerido como parámetro de `check_all_filters`).
- Insertar `check_all_filters(df, asset, trade_log, payout, direction)` con manejo de `auto_shutdown`.
- Logging de cada filtro que bloquee y del auto-shutdown.

### Verificación empírica del histórico
- 0 combinaciones hour+asset con ≥10 trades → filtros dinámicos serán fail-open.
- 3 combinaciones weekday+asset con ≥10 trades, todas WR > 0.55 → sin bloqueo erróneo.

### Tests añadidos (6) — `tests/test_scanner_filters.py`
1. `test_scanner_calls_check_all_filters` — se invoca con parámetros correctos
2. `test_scanner_filter_blocks_before_ml` — ML no se invoca si filtro bloquea
3. `test_scanner_filter_passes_allows_ml` — ML se invoca si filtro pasa
4. `test_scanner_auto_shutdown_stops_scanner` — auto_shutdown setea running=False
5. `test_scanner_no_double_loss_pattern` — loss_pattern_filter no se llama dos veces
6. `test_scanner_payout_calculated_before_filters` — payout se calcula antes de filtros

### Nota arquitectónica para Tarea 1.2
El scanner tiene 3 checkpoints de loop donde evalúa `self.running`. Si `auto_shutdown` activa durante un ciclo, puede ejecutar operaciones adicionales antes de detectar el halt. Mitigación a evaluar en Tarea 1.2.

---

## Tarea 1.1 — Migración Stop Loss y contadores diarios a UTC (2026-04-30)

### Scope original
- Migrar `daily_loss_filter` (`date.today()` → UTC).

### Scope ampliado durante auditoría
- `max_trades_filter` (línea 272) usa la misma función `date.today()` y se migró por coherencia.
- `consecutive_loss_filter` NO usa fecha — no requiere migración (verificado).

### Cambios en `regime_filter.py`
- Constante `SESSION_RESET_TIMEZONE = "UTC"`.
- Helper `_today_utc()`: retorna `datetime.now(timezone.utc).strftime("%Y-%m-%d")`.
- `daily_loss_filter`: `date.today()` → `_today_utc()` + logging UTC en shutdown.
- `max_trades_filter`: `date.today()` → `_today_utc()` + mensaje con "(UTC)".

### Auditoría temporal completa
Todas las referencias a `date.today()` / `datetime.now()` en el repo:

| Archivo | Línea | Uso | Acción |
|---|---|---|---|
| `regime_filter.py:214` | `daily_loss_filter` | Conteo pérdidas | **Migrado a UTC** |
| `regime_filter.py:272` | `max_trades_filter` | Conteo trades | **Migrado a UTC** |
| `regime_filter.py:457` | `check_all_filters` | Ya era UTC | Sin cambio |
| `trader.py:630,672` | Logging | Deprecated | Sin cambio |
| `backtester.py:611,648` | Nombre archivo reporte | No es conteo | Sin cambio |
| `asset_scanner.py` | — | No usa `date.today()` | Verificado limpio |

### Tests añadidos (6)
1. `test_today_utc_helper_returns_iso_format` — formato YYYY-MM-DD sin hora
2. `test_daily_loss_resets_at_utc_midnight` — 23:55 y 00:05 UTC = días distintos
3. `test_daily_loss_uses_utc_not_local_timezone` — servidor UTC-5 no afecta
4. `test_daily_loss_counts_only_utc_today` — solo pérdidas de hoy UTC
5. `test_consecutive_loss_filter_unchanged_by_utc_migration` — TIE invisible preservado
6. `test_max_trades_filter_uses_utc` — conteo de trades usa UTC

### Hallazgos paralelos
- Contaminación dataset ML: 2 trades no-OTC (`EURJPY-op` id=421, `BXY` id=422). Input para Tarea 1.7.

---

## Tarea 0.6 — Deprecación dura de trader/paper_trader/ai_brain (2026-04-30)

### Razón
- Pipelines paralelos con umbrales ML distintos (55% vs 78%).
- ai_brain (LLM como gate de trading) arquitectónicamente cuestionable.
- Carga de mantenimiento que duplicaría trabajo de remediación.

### Mecanismo
- `ALLOW_DEPRECATED_TRADERS = False` por defecto en `iqservice.py`.
- Importación de `trader.py`, `paper_trader.py`, `ai_brain.py` lanza `ImportError` con instrucciones claras.
- `DeprecationWarning` si la flag se activa explícitamente.
- `main.py` usa `_DeprecatedBotStub` inerte cuando `REMEDIATION_MODE=True` (no rompe endpoints legacy).

### Reversibilidad
- Archivos preservados (no borrados).
- Para reactivar: cambiar `ALLOW_DEPRECATED_TRADERS = True` en `iqservice.py` con justificación documentada.

---

## Tarea 0.3 — Auditoría de componentes no documentados (2026-04-30)

### retrain_scheduler.py
- **Estado**: Estaba activo dentro de `main.py --mode paper` (PID 23441). Nunca completó un ciclo de reentrenamiento (`retrain_state.json` no existe, `models/versions/` vacío).
- **Acción**: Desactivado durante remediación. En `main.py`, si `REMEDIATION_MODE=True`, no se invoca `retrain_scheduler.start()`.
- **Riesgo detectado**: `fetch_training_data()` no filtra por OTC — incluiría trades no-OTC si existieran.

### ai_brain.py (LLM fallback)
- **Uso**: Solo invocado por `trader.py` (bot single-asset) cuando ML no está cargado.
- **No usado por** `asset_scanner.py` (motor principal). El scanner requiere ML obligatoriamente.
- **Estado**: Legacy. No afecta operaciones actuales.

### paper_trader.py vs asset_scanner.py
- Pipelines de decisión divergentes: scanner usa cascada de 4 estrategias + ML obligatorio (55%), trader usa ML/LLM con doble gate (78%).
- `asset_scanner` es el motor principal en uso.
- **Decisión de deprecación**: Pendiente de aprobación (Opción 1 recomendada).

### strategy_config.yaml
- Umbrales catalogados. BB 2-Candle hardcodeado en `indicators.py`, no en YAML.
- Cambios para Tarea 2.2 requieren edición en **ambos** (YAML y código).

### min_streak_filter
- Solo aplica cuando `is_extreme=True` — no afecta señales de BB o RSI clásico.
- Redundancia parcial con `pre_qualify()` (2 velas mínimo vs 3 en filtro). Consolidar en Tarea 2.2.

### Proceso bot
- PID 23441 (`main.py --mode paper`): detenido (auto-shutdown o cierre manual).
- 118 trades generados (2026-04-23 a 2026-04-30). Distribución: 59W / 55L / 3T / 1P.
- Modelo ML verificado sin modificación post-shutdown (hash = baseline `f1709cb4...`).

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
