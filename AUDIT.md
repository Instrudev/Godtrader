# AUDIT.md — Fase 0: Auditoría del Repositorio

**Fecha:** 2026-04-20  
**Estado:** Borrador para aprobación antes de codificar

---

## 1. Mapa de archivos y responsabilidades

| Archivo | Responsabilidad real |
|---|---|
| `main.py` | Servidor FastAPI. Expone endpoints REST (`/connect`, `/bot/start`, `/bot/stop`, `/bot/status`) y dos WebSockets (`/ws/notifications`, `/ws/{asset_id}`). Singleton del bot en módulo-nivel. |
| `trader.py` | Orquestador del bot. Clase `TradingBot` con bucle asyncio, filtro de pre-calificación, doble gatillo GATE-0/1/2, ejecución de órdenes, monitoreo de resultado, log de pérdidas. |
| `ai_brain.py` | Motor de decisión LLM. Construye snapshot de 3 bloques (MACRO/TECH/MICRO) y lo envía a Ollama. Devuelve `{op, pr, ex, an}`. |
| `indicators.py` | Cálculo de indicadores técnicos: EMA, RSI, Bollinger, VolRel, soporte/resistencia, patrones de vela, índice de adherencia, VSA. No tiene dependencias TA externas (solo numpy/pandas). |
| `iqservice.py` | Wrapper del cliente `iqoptionapi`. Clase `Exnova_Option` que monkey-patchea URLs de IQ Option hacia `trade.exnova.com`. Método `connect()`, datos históricos, stream en tiempo real, órdenes binary/digital. |
| `iqoption.py` | Script standalone de prueba de conexión a IQ Option puro (no Exnova). **Obsoleto.** |
| `test_conexion.py` | Script de prueba manual con credenciales hardcodeadas. **No es un test automatizado.** |
| `trades_history.json` | Log de las últimas 15 **pérdidas** (no historial completo). Nunca registra wins. |
| `static/` | Frontend HTML/CSS/JS (index.html, style.css, app.js). No auditado en detalle. |
| `README.md` | Documentación de instalación y uso. Contiene valores inconsistentes con el código real (ver sección 6). |
| `.env` | Variables de entorno: credenciales del broker, modelo Ollama. **Contiene credenciales reales — NO commitear.** |

---

## 2. Pipeline real del ciclo de 60s

```
cada 5s (LOOP_POLL_SECS)
│
├─ ¿current_ts > last_ts?  NO → esperar
│
└─ SÍ: nueva vela cerrada
    │
    ├─ ¿cooldown activo? SÍ → skip
    │
    ├─ get_candles(asset, 60s, 300) via iqservice
    │   └─ Si < 205 velas → skip
    │
    ├─ build_dataframe() → EMA20, EMA200, RSI14, BB(20,2), VolRel
    │
    ├─ pre_qualify()
    │   ├─ RSI < 35 o RSI > 65  (no 30/70 como dice el README)
    │   └─ pct_b <= 0.15 o pct_b >= 0.85
    │   └─ Falla → skip silencioso
    │
    ├─ get_ai_decision() → Ollama (phi3 por defecto, no llama3:8b)
    │   └─ Construye 6 bloques: CICLO, MACRO, TECH, MICRO, HISTORIAL_ERRORES, SENTIMIENTO_BROKER
    │   └─ Parsea JSON {op, pr, ex, an}
    │
    ├─ GATE-0: op == "WAIT" → no operar
    ├─ GATE-1: pr < 78 → no operar  (no 85% como dice README/main.py)
    ├─ GATE-2: dirección vs EMA200 OR EMA20 (lógica OR, no AND estricto)
    │
    └─ _place_order() → api.buy() o api.buy_digital_spot()
        └─ cooldown = ex × 60s + 10s
        └─ lanzar _monitor_trade() en background (sin timeout)
```

---

## 3. Manejo de estado

| Aspecto | Implementación actual |
|---|---|
| **Cooldown** | `_cooldown_until: float` (epoch time). Se saltea el análisis completo mientras esté activo. |
| **Memoria de errores** | `trades_history.json`: últimas 15 pérdidas en disco (JSON). Solo losses, no wins. |
| **Memoria en sesión** | `signal_history`: últimas 5 señales en RAM (se pierde al reiniciar). `trade_log`: log completo en RAM (también se pierde). |
| **Conexión Exnova** | WebSocket persistente gestionado por `iqoptionapi`. `is_connected()` hace check + reconexión automática si cae. No hay backoff exponencial. |
| **Estado del bot** | Singleton `trading_bot` en módulo-nivel. No hay persistencia de estado entre reinicios del servidor. |

---

## 4. Conexión a Exnova (Broker Integration)

**Método:** La librería `iqoptionapi` (cliente no oficial de IQ Option, open source) es monkey-patcheada en tiempo de ejecución para redirigir todas las URLs al dominio de Exnova.

**Flujo de conexión:**

1. Se instancia `Exnova_Option(email, password)` que hereda de `IQ_Option`.
2. En `connect()`, se crea un nuevo `IQOptionAPI("trade.exnova.com", ...)`.
3. Se sobreescribe `wss_url = "wss://ws.trade.exnova.com/echo/websocket"`.
4. Se monkey-patchea `send_http_request_v2` para reemplazar dominios en cada petición HTTP:
   - `auth.iqoption.com/api` → `api.trade.exnova.com`
   - `*.iqoption.com` → `*.trade.exnova.com`
5. Para el login, se inyectan headers `Origin` y `Referer` de Exnova.

**Endpoints usados:**
- HTTP REST: `https://api.trade.exnova.com/v2/login` (autenticación)
- WebSocket: `wss://ws.trade.exnova.com/echo/websocket` (stream de precios y órdenes)
- Velas históricas: via WebSocket (protocolo propietario de IQ Option)

**Puntos de fragilidad:**
- La librería `iqoptionapi` no tiene mantenimiento activo y asume la API de IQ Option, que puede divergir de Exnova.
- El monkey-patch es frágil: cualquier cambio interno en `iqoptionapi` puede romperlo sin aviso.
- No hay SDK oficial de Exnova. El contrato de API es implícito y no documentado.
- `_monitor_trade` usa `api.get_async_order()` con **bucle while True sin timeout**: si la orden no cierra (error del broker), el monitor queda colgado indefinidamente, consumiendo un asyncio Task de forma permanente.
- La reconexión automática en `is_connected()` no tiene backoff: puede generar flood de reconexiones si el broker cae.

---

## 5. Tests existentes

**Ninguno automatizado.**

- `test_conexion.py`: Script manual de prueba de orden. Tiene credenciales hardcodeadas (email + password en texto plano en el código). No es un test unitario ni de integración ejecutable por CI.
- `iqoption.py`: Script de prueba de conexión a IQ Option puro. Obsoleto y redundante.
- No hay `pytest`, `unittest`, ni configuración de CI (GitHub Actions o similar).

**Cobertura de tests: 0%**

---

## 6. Inconsistencias entre documentación y código real

Estas son las desviaciones encontradas, ordenadas por impacto:

### 6.1 MIN_PROBABILITY: 78% real vs 85% documentado

| Fuente | Valor |
|---|---|
| `trader.py` (código ejecutado) | `MIN_PROBABILITY = 78` |
| `README.md` tabla | 85% |
| `README.md` diagrama | "pr ≥ 85%" |
| `main.py` docstring de `/bot/start` | "pr ≥ 85% AND dirección == EMA200" |
| `main.py` respuesta `/bot/start` | `"min_probability": 78` ← este sí es correcto |
| `ai_brain.py` system prompt R9 | "pr<78 → op=WAIT" ← consistente con 78 |

**Impacto:** El bot opera con umbral más permisivo (78%) del que se comunica al usuario (85%). El LLM también aplica R9 internamente, lo que significa que el LLM puede decidir WAIT con pr=79 pero el código lo dejaría pasar si el LLM dijera CALL con pr=79 ignorando su propia regla.

### 6.2 Modelo Ollama: phi3 real vs llama3:8b documentado

| Fuente | Modelo |
|---|---|
| `ai_brain.py` default (`OLLAMA_MODEL`) | `phi3` |
| `main.py` FastAPI description | "phi3 via Ollama" |
| `main.py` docstring `/bot/start` | "phi3 local" |
| `trader.py` banner en logs | "llama3:8b" |
| `README.md` (producción recomendado) | "llama3:8b" |

**Impacto:** El modelo que arranca por defecto (phi3) es distinto al que aparece en los logs del bot ("llama3:8b") y al que el README presenta como modelo de producción. Esto dificulta la reproducibilidad.

### 6.3 GATE-2: OR lógico vs AND documentado

- **Documentado:** "dirección debe coincidir con tendencia EMA200" (filtro fuerte, requiere alineación macro).
- **Implementado:** `valid_call = (macro_trend == "BULLISH" **or** micro_trend == "BULLISH")`. Basta con que EMA20 (periodo corto) esté alineado para aprobar el gate. Esto hace el filtro muy permisivo: permite operar contra EMA200 si EMA20 está a favor.

**Impacto:** El filtro de tendencia macro es prácticamente inefectivo. En mercado lateral, EMA20 cambia de lado cada pocas velas, aprobando casi todas las señales.

### 6.4 Umbrales RSI del pre-filtro: 35/65 real vs 30/70 en README

- `indicators.py`: `rsi < 35 or rsi > 65`
- `README.md` tabla de parámetros: "RSI < 30 / > 70"

**Impacto:** El bot activa el análisis con más frecuencia de lo que el README indica.

### 6.5 `bb_pos` calculado incorrectamente en el log de pérdidas

En `trader.py._save_loss_record()`:
```python
"bb_pos": (candle.get("close", 0) - candle.get("bb_lower", 0)) / (candle.get("bb_width", 1) or 1)
```
`bb_width` es el **porcentaje** `(upper - lower) / mid * 100`, no el rango absoluto en precio. El denominador debería ser `(bb_upper - bb_lower)`. El campo guardado en `trades_history.json` es incorrecto y no es comparable con el `pct_b` calculado en `pre_qualify()`. Visible en los datos: bb_pos tiene valores como 1.80, 2.11 (>1.0), lo que confirma el error.

### 6.6 `vol_rel` siempre 1.0 en registros de pérdidas

Todos los registros en `trades_history.json` muestran `vol_rel: 1.0`. Los pares OTC no tienen volumen real; el fallback proxy (`high - low`) produce valores que `_relative_volume()` termina normalizando a 1.0 porque la media móvil del proxy también sube proporcionalmente. **El indicador de volumen relativo no aporta señal real en OTC.**

### 6.7 Credenciales en texto plano en test_conexion.py

`test_conexion.py` líneas 5-6 contienen email y contraseña reales hardcodeadas. **Riesgo crítico de seguridad**: si el archivo se commitea (ya pudo haberse commitado), las credenciales quedan expuestas en el historial de git.

**Acción inmediata recomendada:** verificar que `.gitignore` excluye `test_conexion.py` o limpiar las credenciales del archivo.

### 6.8 `_monitor_trade` sin timeout (bucle infinito potencial)

```python
while True:
    data = await asyncio.to_thread(iq_service.api.get_async_order, order_id)
    ...
    await asyncio.sleep(1.0)
```
No hay condición de salida si la orden no cierra. Cada operación ejecutada crea un Task asyncio que puede quedar activo indefinidamente si hay un error del broker. Con el tiempo, esto agota el event loop.

### 6.9 `trades_history.json` solo registra pérdidas

El nombre `trades_history` sugiere historial completo, pero el método `_save_loss_record` solo se llama cuando `profit < 0`. Las victorias no se registran nunca. Esto hace imposible calcular winrate histórico real sin correr el bot en tiempo real.

### 6.10 `ai_brain.py` tiene 4 bloques adicionales no mencionados en README

El README describe 3 bloques (MACRO, TECH, MICRO). El código construye 6: CICLO, MACRO, TECH, MICRO, HISTORIAL_ERRORES, SENTIMIENTO_BROKER. El bloque SENTIMIENTO_BROKER combina `calculate_adherence_index` y `detect_vsa_anomaly`.

---

## 7. Resumen de deuda técnica pre-refactor

| Categoría | Problema | Impacto |
|---|---|---|
| Arquitectura | LLM como motor de decisión (no determinista, sin calibración) | Alto |
| Arquitectura | sin tests automatizados | Alto |
| Arquitectura | sin backtester ni forward tester | Alto |
| Arquitectura | sin filtros de régimen OTC | Alto |
| Datos | vol_rel inútil en OTC (siempre 1.0) | Medio |
| Datos | bb_pos calculado incorrectamente en logs | Medio |
| Seguridad | Credenciales hardcodeadas en test_conexion.py | Alto |
| Consistencia | MIN_PROBABILITY 78 vs 85 en README | Medio |
| Consistencia | Modelo phi3 vs llama3:8b en documentación | Bajo |
| Consistencia | GATE-2 OR vs AND documentado | Medio |
| Consistencia | RSI 35/65 vs 30/70 en README | Bajo |
| Robustez | _monitor_trade sin timeout | Medio |
| Robustez | Sin backoff en reconexión | Bajo |
| Mantenibilidad | iqoption.py obsoleto | Bajo |

---

## 8. Plan de migración propuesto (por fase)

### Fase 0 (este documento) — Auditoría completa ✓

### Fase 1 — Infraestructura de validación
- Crear `backtester.py`: replay vela a vela sin look-ahead bias
- Crear `paper_trader.py`: opera igual que bot real pero en demo, sin órdenes reales
- Crear `trades.db` (SQLite): esquema completo con features, probabilidad, resultado, payout
- Demostrar baseline del bot ACTUAL en backtest (sin cambiar lógica)
- Limpiar credenciales de `test_conexion.py`

### Fase 2 — Filtros de régimen OTC
- Crear `regime_filter.py`: hour_profile, weekday_profile, volatility, payout, daily_loss, consecutive_loss, max_trades, drift
- Crear `generator_drift_detector.py`: monitoreo estadístico semanal del feed OTC
- Integrar antes del clasificador en el pipeline
- Backtest comparativo vs baseline

### Fase 3 — Estrategia base adaptada a OTC
- Reescribir `indicators.py`: half-life mean reversion (O-U test), heatmap horario/día, rachas, ATR relativo
- Eliminar filtro EMA200 como tendencia de mercado real
- Crear `strategy_config.yaml` con perfiles por par y franja
- Backtest comparativo

### Fase 4 — Clasificador ML
- Crear `ml_classifier.py`: LightGBM + calibración Platt/isotonic
- Crear `train_model.py`: entrena desde CSV histórico etiquetado
- Desacoplar `ai_brain.py` del path crítico (mantener como módulo opcional)
- Backtest comparativo

### Fase 5 — Loop de reentrenamiento
- Crear `retrain_scheduler.py`: reentrenamiento periódico con criterio de promoción por Profit Factor
- Versionado de modelos en `models/` con metadata

### Fase 6 — Documentación final
- README actualizado con riesgos OTC explícitos
- `BROKER_INTEGRATION.md` con detalles de conexión a Exnova
- Diagrama de arquitectura nuevo
- GitHub Actions CI

---

## Decisiones de diseño que requieren tu aprobación antes de codificar

1. **SQLite vs archivo JSON**: El plan dice SQLite para `trades.db`. ¿Confirmas que podemos instalar `sqlite3` (ya viene con Python) y eventualmente `aiosqlite` para acceso async?

2. **LightGBM vs XGBoost**: El prompt menciona ambos. Para Apple Silicon con Metal, LightGBM tiene mejor soporte nativo. ¿Confirmo LightGBM como primera opción?

3. **Backtester con data histórica**: Para Fase 1, el backtester necesita datos históricos OTC. ¿Tienes CSV de velas históricas exportado de Exnova, o debemos construir el dataset grabando desde el bot en demo durante unos días primero?

4. **`ai_brain.py` en producción actual**: ¿Dejo el LLM completamente desconectado del pipeline desde Fase 1, o lo mantengo activo hasta que el clasificador ML esté validado en Fase 4?

5. **Credenciales en `test_conexion.py`**: ¿Autorizo eliminar/limpiar las credenciales hardcodeadas de ese archivo ahora mismo, antes de arrancar las fases?

---

*Esperando aprobación del plan antes de comenzar Fase 1.*
