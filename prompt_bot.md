

## CONTEXTO DEL PROYECTO

Trabajas sobre un bot autónomo de trading de opciones binarias que opera en **Exnova, mercado OTC, velas de 1 minuto**. La arquitectura actual está en Python e incluye, como mínimo, los siguientes módulos:

- `indicators.py` — cálculo de indicadores técnicos y filtro de pre-calificación
- `ai_brain.py` — consulta a un LLM local (Ollama, llama3:8b) para decisión
- `trader.py` — sistema de gatillos (GATE-0, GATE-1, GATE-2) y ejecución
- `trades_history.json` — log de últimos 15 errores
- Notificaciones WebSocket al frontend (eventos ANALYSIS, SIGNAL, EXECUTION, ERROR)

### CONTEXTO CRÍTICO — Mercado OTC en Exnova

Esto **no** es forex regulado. El feed de precios OTC de Exnova es generado internamente por el broker, no proviene de un consenso de mercado real. Esto cambia varias suposiciones de diseño:

- **No hay calendario económico relevante.** Las noticias macro no mueven el precio OTC. El generador interno del broker controla el feed.
- **El mercado OTC opera 24/7**, incluyendo fines de semana, cuando los pares forex reales están cerrados.
- **El algoritmo del broker puede cambiar sin aviso.** Cualquier estrategia rentable hoy puede dejar de serlo mañana si Exnova ajusta parámetros del PRNG o de la simulación de volatilidad. El bot debe detectar este cambio automáticamente y apagarse.
- **El backtesting tiene valor limitado**: data histórica OTC describe el algoritmo pasado, no el actual. El forward test es más importante que el backtest en este contexto.
- **El edge real en OTC viene de detectar regularidades del generador**: periodicidades horarias, sesgos por par, patrones de mean reversion forzada. NO de análisis técnico clásico, que asume comportamiento de mercado real.

Antes de tocar código, **lee el repositorio completo** y construye un mapa mental de:

1. Estructura de archivos y sus responsabilidades
2. Pipeline real de cada ciclo de 60s
3. Manejo de estado (cooldown, memoria, conexión al broker Exnova)
4. Cliente API o método de conexión usado para Exnova (no hay SDK oficial maduro; documenta cómo se conecta)
5. Tests existentes (si los hay)
6. Cualquier desviación entre la lógica documentada y el código real

Si encuentras inconsistencias, repórtalas antes de modificar nada.

---

## OBJETIVO GENERAL

Refactorizar el bot para corregir cinco problemas estructurales identificados por análisis cuantitativo previo. **No se trata de optimizar parámetros**: hay decisiones de arquitectura que deben cambiar. El objetivo final es un bot con edge estadístico real, validable mediante backtesting riguroso, no un bot que "parece inteligente".

---

## PROBLEMAS A RESOLVER (ordenados por impacto en rentabilidad)

### Problema 1 — Reemplazar el LLM como motor de decisión

El uso de llama3:8b como cerebro de decisión es estructuralmente inadecuado: un LLM generalista no produce probabilidades calibradas, es no determinista (mismo setup → respuesta distinta), y no aprende de los resultados reales del bot. El campo `pr` que devuelve es ruido, no probabilidad.

**Acción requerida:**

- Eliminar `ai_brain.py` como dependencia obligatoria del pipeline (mantenerlo como módulo opcional para análisis cualitativo, no para decisión).
- Crear `ml_classifier.py` con un modelo gradient boosting (LightGBM o XGBoost) que reciba un vector de features y devuelva una probabilidad calibrada de éxito de la operación.
- Implementar calibración de probabilidades (Platt scaling o isotonic regression) usando un set de validación separado.
- El nuevo modelo debe exponer una interfaz mínima: `predict_proba(features: dict) -> {"call_proba": float, "put_proba": float}`.
- Definir el feature set inicial **adaptado a OTC**: RSI(14), distancia normalizada a banda BB, ATR(14) normalizado contra ATR de últimas 24h, hora UTC, día de la semana, longitud de racha direccional actual, retorno acumulado últimas N velas (3, 5, 10), autocorrelación de retornos en ventana corta, half-life de mean reversion estimada, payout actual del par, winrate histórico del par en esa franja horaria.
- Incluir script `train_model.py` que entrena desde un CSV histórico de operaciones etiquetadas (win/loss) y guarda el modelo serializado en `models/`.

### Problema 2 — Rediseñar la estrategia base para OTC (no forex real)

La combinación "RSI extremo + toque BB" es una estrategia pensada para comportamiento de mercado real. En OTC, el edge no viene del análisis técnico clásico sino de **detectar regularidades del generador de precios del broker**. Hay que reorientar la estrategia a lo que realmente funciona en OTC.

**Acción requerida:**

- Reescribir `indicators.py` con un enfoque orientado a OTC:
  - **Detección de mean reversion forzada**: muchos generadores OTC exhiben reversión a la media más fuerte que el forex real. Medir half-life de reversión por par con test de Ornstein-Uhlenbeck sobre data histórica; operar solo pares con half-life corta y consistente.
  - **Perfil por hora del día y día de la semana**: crear un "heatmap" de winrate histórico del setup por franja horaria y día. Muchos brokers OTC tienen sesgos por horario (ej. volatilidad artificial más alta en ciertas horas de Asia).
  - **Detección de rachas**: medir longitud media de rachas direccionales por par; cuando una racha excede el percentil 90 histórico, operar reversión.
  - **Volatilidad relativa**: ATR M1 normalizado contra su media de las últimas 24h, no un umbral absoluto.
  - Descartar completamente el filtro de "EMA200 H1" como tendencia de mercado real: en OTC no aplica el mismo concepto.
- Confirmación con price action (pin bar, engulfing) **solo como filtro secundario**, no como señal principal.
- Permitir configurar la estrategia desde un archivo `strategy_config.yaml` con perfiles por par y por franja horaria.
- Añadir un módulo `generator_drift_detector.py` que monitoree semanalmente la estabilidad estadística del feed OTC (media, varianza, autocorrelación de retornos, test KS contra ventana previa). Si detecta un cambio estadísticamente significativo, el bot se apaga automáticamente y notifica: probablemente Exnova cambió su algoritmo.

### Problema 3 — Filtros de régimen adaptados a OTC

Los filtros tradicionales de noticias y sesiones no aplican en OTC. Los filtros que sí aportan en OTC son distintos.

**Acción requerida:**

Crear `regime_filter.py` con los siguientes módulos, cada uno como función pura que devuelve `(allow: bool, reason: str)`:

- `hour_profile_filter()` — bloquea operaciones en franjas horarias donde el winrate histórico del par esté por debajo del 52% (umbral configurable). Reemplaza el filtro de sesión forex tradicional.
- `weekday_profile_filter()` — análogo al anterior pero por día de la semana (los OTC suelen comportarse distinto sábado/domingo vs días hábiles).
- `volatility_filter()` — bloquea cuando ATR percentil < 30 o > 95 sobre ventana móvil de 7 días (no 30; el algoritmo del broker puede cambiar).
- `payout_filter()` — bloquea si el payout actual del par cae bajo un umbral mínimo (default 0.80). Exnova reduce payouts dinámicamente; operar con payout 0.70 destruye edge.
- `daily_loss_filter()` — apaga el bot por el resto del día si las pérdidas superan X% del capital (default 5%).
- `consecutive_loss_filter()` — apaga el bot tras N pérdidas consecutivas (default 3); puede indicar drift del generador.
- `max_trades_filter()` — limita a N operaciones por día (default 5).
- `drift_filter()` — consulta a `generator_drift_detector.py`; si hay drift detectado en los últimos 7 días, no operar.

Integrar estos filtros antes del clasificador ML en el pipeline.

**Nota:** NO incluir filtro de noticias macro ni de sesiones forex tradicionales. No aplican al OTC y añadir lógica innecesaria solo introduce bugs.

### Problema 4 — Sustituir la "memoria" de 15 errores por aprendizaje real

El log actual no modifica el comportamiento del bot. Necesitamos un loop de aprendizaje real: cada operación cerrada se etiqueta y alimenta al dataset de entrenamiento, y el modelo se reentrena periódicamente.

**Acción requerida:**

- Reemplazar `trades_history.json` por una base de datos SQLite (`trades.db`) con esquema completo: timestamp, par, dirección, expiry, features completas en el momento de la entrada, probabilidad predicha, resultado real, payout, pips de diferencia.
- Crear `retrain_scheduler.py` que cada N días (configurable, default 7) reentrena el modelo con los datos acumulados, valida contra el periodo más reciente y solo promueve el nuevo modelo si su Profit Factor en validación supera al modelo en producción.
- Mantener versionado de modelos en `models/` con metadata (fecha entrenamiento, métricas, número de operaciones de entrenamiento).

### Problema 5 — Validación: forward test prevalece sobre backtest en OTC

En OTC, un buen backtest NO garantiza rentabilidad futura porque el broker puede cambiar el algoritmo. Por eso el forward test en cuenta demo es el criterio definitivo de aceptación.

**Acción requerida:**

- Crear `backtester.py` que:
  - Acepte un rango de fechas y un par
  - Reproduzca la lógica completa del bot vela a vela (sin look-ahead bias)
  - Aplique el payout real histórico del broker (no asumir payout fijo)
  - Genere reporte con: número de trades, winrate, Profit Factor, Sharpe, max drawdown, distribución de retornos por mes, equity curve, winrate por hora y por día de semana
- Crear `paper_trader.py` para forward test sobre cuenta demo de Exnova: opera exactamente como el bot real pero sin enviar órdenes con dinero real. Registra todo en la misma SQLite.
- **Criterio de aceptación para despliegue a real:**
  - Backtest sobre 3 meses de data OTC debe cumplir: Profit Factor ≥ 1.4, winrate ≥ 60% (con payout 0.80), max drawdown ≤ 15%.
  - **Y además**, forward test en demo durante mínimo 30 días consecutivos con métricas no peores que el 85% de las del backtest.
  - Si el forward test diverge significativamente del backtest, el algoritmo del broker probablemente cambió: NO desplegar a real.

---

## CONSTRAINTS TÉCNICOS

- Mantener compatibilidad con el cliente actual de Exnova (no cambiar el método de conexión al broker salvo que el actual esté roto o sea inseguro).
- Documentar en `BROKER_INTEGRATION.md` cómo se conecta el bot a Exnova (API, WebSocket, scraping, etc.), qué endpoints usa y cuáles son los puntos de fragilidad conocidos.
- Mantener el sistema de notificaciones WebSocket al frontend, pero añadir nuevos eventos: `REGIME_BLOCK`, `MODEL_RETRAIN`, `BACKTEST_REPORT`, `DRIFT_DETECTED`, `BOT_AUTOSHUTDOWN`.
- El bot debe poder operar en modo `live`, `paper` o `backtest` mediante flag de CLI.
- Optimizar para el hardware actual: Apple Silicon con 16GB de RAM unificada. LightGBM cabe holgado; evita modelos pesados.
- Tipado estricto con `from __future__ import annotations` y type hints en todas las funciones públicas.
- Logs estructurados (JSON) en `logs/` rotados por día.
- Cobertura de tests mínima: 70% en módulos nuevos, con tests unitarios para cada filtro y para el clasificador.

---

## ENTREGABLES POR FASE

Trabaja en fases verificables. **No avances a la siguiente fase sin que la anterior pase tests y backtest.**

**Fase 0 — Auditoría:** Reporte en `AUDIT.md` con el mapa del repo, inconsistencias encontradas y plan de migración detallado. Pide aprobación antes de codificar.

**Fase 1 — Infraestructura de validación:** `backtester.py`, `paper_trader.py`, esquema SQLite, dataset histórico inicial. Demostrar que el bot ACTUAL (sin cambios) corre en backtest y reportar sus métricas reales como baseline.

**Fase 2 — Filtros de régimen:** `regime_filter.py` integrado, con tests. Re-correr backtest y comparar contra baseline.

**Fase 3 — Estrategia base mejorada:** Nuevo `indicators.py` con multi-timeframe y price action. Backtest comparativo.

**Fase 4 — Clasificador ML:** `ml_classifier.py`, `train_model.py`, modelo entrenado y calibrado. Backtest comparativo.

**Fase 5 — Loop de reentrenamiento:** `retrain_scheduler.py`, versionado de modelos, criterio de promoción.

**Fase 6 — Documentación final:** README actualizado, diagrama de arquitectura nuevo, runbook operacional.

---

## CRITERIOS DE ACEPTACIÓN GLOBALES

Al terminar el refactor, el bot debe cumplir:

1. Cero llamadas al LLM en el path crítico de decisión.
2. Backtest reproducible sobre data OTC histórica con métricas documentadas.
3. Forward test de al menos 30 días en demo, con métricas alineadas al backtest.
4. Toda decisión de operar es trazable (qué filtros pasaron, qué probabilidad asignó el modelo, qué features se usaron).
5. Apagado automático ante pérdida diaria máxima, racha de pérdidas consecutivas, drift del generador OTC detectado, o payout bajo umbral.
6. Tests pasando en CI (configura GitHub Actions si no existe).
7. Documentación explícita en README sobre los riesgos del OTC y el hecho de que el edge puede desaparecer si Exnova cambia su algoritmo.

---

## QUÉ NO HACER

- No introducir martingala ni grid trading bajo ninguna forma.
- No usar APIs de pago o servicios cloud sin pedir aprobación previa.
- No modificar el frontend WebSocket más allá de añadir los nuevos tipos de evento.
- No commitear claves de API, credenciales del broker ni datos históricos pesados (>50MB) al repo; usa `.env` y `.gitignore`.
- No optimizar parámetros sobre el set de test (eso es overfitting; usa walk-forward analysis).

---

## FORMATO DE TRABAJO ESPERADO

Para cada fase:

1. Pide confirmación del plan antes de codificar.
2. Implementa con commits pequeños y descriptivos.
3. Al terminar la fase, entrega un reporte breve con: archivos creados/modificados, tests añadidos, resultados de backtest comparativo y próximos pasos.
4. Si encuentras una decisión de diseño con trade-offs reales, pregunta antes de elegir por mí.

Empieza por la Fase 0.
