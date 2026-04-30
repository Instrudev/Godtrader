# Changelog — Branch remediation/v1

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
