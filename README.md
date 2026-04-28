# Exnova OTC Trading Bot — ML Edition

Bot de trading autónomo para **opciones binarias en mercado OTC de Exnova**.  
Motor de decisión: **LightGBM + Platt scaling** (calibración probabilística).  
Entrenamiento automático sobre datos acumulados en producción (paper trading).

> **ADVERTENCIA CRÍTICA:** Este bot opera en mercado OTC (Over-The-Counter) de Exnova,
> cuyo feed de precios es generado internamente por el broker mediante un PRNG privado.
> **No es mercado real de divisas.** El edge estadístico detectado puede desaparecer en
> cualquier momento si Exnova modifica su algoritmo generador de precios. Leer la sección
> [Riesgos OTC](#riesgos-otc) antes de operar con dinero real.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     CICLO POR VELA (60s)                        │
│                                                                 │
│  Exnova API (iqoptionapi monkey-patched)                        │
│       │ 300 velas OHLCV M1                                      │
│       ▼                                                         │
│  indicators.py  ─── build_dataframe() ───────────────────────  │
│    EMA20/200 · RSI14 · Bollinger(20,2) · ATR(14)               │
│    VolRel · Racha direccional · Half-life OU                    │
│       │                                                         │
│       ▼                                                         │
│  regime_filter.py ── 8 filtros OTC ──────────────────────────  │
│    ✗ payout < 80%     → BLOCK                                   │
│    ✗ 3 pérdidas día   → AUTOSHUTDOWN                           │
│    ✗ 3 pérdidas consec → AUTOSHUTDOWN                          │
│    ✗ drift generador  → AUTOSHUTDOWN                           │
│    ✗ ATR fuera rango  → BLOCK                                   │
│       │ Todos OK                                                │
│       ▼                                                         │
│  pre_qualify() ─── Filtro OTC primario ──────────────────────  │
│    Racha extrema (percentil > 70%) → reversión esperada         │
│    Fallback: RSI extremo + toque BB                             │
│       │ Califica                                                │
│       ▼                                                         │
│  MLClassifier (LightGBM + Platt)  ←── models/lgbm_model.pkl   │
│    17 features: RSI, BB, ATR, racha, retornos, autocorr, HL…   │
│    → call_proba / put_proba                                     │
│    [Sin modelo: fallback LLM Ollama]                            │
│       │                                                         │
│       ▼                                                         │
│  GATE-1: proba ≥ 78%   ─── Falla ───► No operar               │
│  GATE-2: dirección coherente con racha OTC extrema              │
│       │ Ambos superados                                         │
│       ▼                                                         │
│  api.buy() → Cooldown (ex × 60s + 10s)                         │
│  paper_trader: simula + registra en trades.db                   │
│       │                                                         │
│       ▼  (background, cada 6h)                                  │
│  retrain_scheduler.py                                           │
│    ¿≥ 50 trades nuevos? → reentrenar LightGBM                  │
│    challenger PF > champion PF → PROMOVER                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Riesgos OTC

**Leer obligatoriamente antes de operar con dinero real.**

### 1. El precio es sintético
Exnova OTC genera precios internamente mediante un algoritmo PRNG privado.
No existe un mercado subyacente real. El bot no analiza forex real;
detecta regularidades estadísticas del generador del broker.

### 2. El edge puede desaparecer sin aviso
Si Exnova modifica su algoritmo (lo hace periódicamente), el edge detectado
desaparece inmediatamente. El módulo `generator_drift_detector.py` monitorea
cambios estadísticos y apaga el bot automáticamente cuando los detecta,
pero solo puede reaccionar después del cambio, no anticiparlo.

### 3. El backtest no garantiza rentabilidad futura
Un buen backtest sobre datos históricos OTC no implica rentabilidad futura
porque el broker puede cambiar el algoritmo entre el periodo de entrenamiento
y el de operación. El forward test en cuenta demo (paper trading) es el
criterio definitivo de aceptación, no el backtest.

### 4. Riesgo de pérdida total
Las opciones binarias son instrumentos de alto riesgo. Es posible perder
todo el capital operado. El bot incluye salvaguardas (máximo 3 pérdidas
diarias, 3 consecutivas, payout mínimo 80%), pero no elimina el riesgo.

### 5. Regulación
Las opciones binarias están prohibidas o reguladas en muchos países.
Verifica la legalidad en tu jurisdicción antes de operar.

---

## Instalación

```bash
# 1. Clonar y crear entorno virtual
git clone <repo>
cd "analisis mercados"
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install fastapi uvicorn numpy pandas pyyaml python-dotenv \
            lightgbm scikit-learn iqoptionapi ollama

# 3. macOS: dependencia nativa de LightGBM
brew install libomp

# 4. Configurar credenciales
cp .env.example .env
# Editar .env con email/password de Exnova

# 5. (Opcional) Ollama como fallback LLM
brew install ollama
ollama pull phi3
```

---

## Modos de operación

### Paper trading (recomendado para empezar)
Opera con precios reales pero sin enviar órdenes. Acumula datos en `trades.db`.

```bash
python main.py --mode paper
# → http://localhost:8000
```

### Live trading
Opera con órdenes reales en la cuenta activa del broker.

```bash
python main.py --mode live
```

### Backtest (cuando haya datos acumulados)
```bash
python backtester.py run --csv data/EURUSD-OTC.csv --asset EURUSD-OTC
```

### Entrenar el clasificador ML
```bash
# Verificar que hay suficientes trades en trades.db (mínimo 50)
python train_model.py

# Forzar reentrenamiento sin condiciones
python retrain_scheduler.py --force

# Ver estado del scheduler en producción
curl http://localhost:8000/retrain/status | python3 -m json.tool
```

---

## Flujo de arranque completo

```bash
# Terminal 1: iniciar el servidor
python main.py --mode paper

# Terminal 2: conectar al broker
curl -X POST http://localhost:8000/connect \
  -d "email=tu@email.com&password=tupass"

# Iniciar el bot (acumular datos en demo)
curl -X POST http://localhost:8000/bot/start \
  -H "Content-Type: application/json" \
  -d '{"asset": "EURUSD-OTC", "asset_type": "binary", "amount": 1.0}'

# Monitorear
curl http://localhost:8000/bot/status | python3 -m json.tool

# Después de ≥ 50 trades cerrados: entrenar modelo
python train_model.py

# Detener bot
curl -X POST http://localhost:8000/bot/stop
```

---

## Endpoints API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/connect` | Autenticar con Exnova |
| `GET`  | `/assets` | Listar activos disponibles |
| `GET`  | `/candles/{asset}` | Histórico de 300 velas |
| `POST` | `/bot/start` | Iniciar bot |
| `POST` | `/bot/stop` | Detener bot |
| `GET`  | `/bot/status` | Estado completo + estadísticas |
| `GET`  | `/retrain/status` | Estado del scheduler ML |
| `POST` | `/retrain/trigger` | Forzar reentrenamiento (respeta condición de trades) |
| `POST` | `/retrain/force` | Forzar reentrenamiento sin condiciones |
| `WS`   | `/ws/notifications` | Stream de eventos del bot |
| `WS`   | `/ws/{asset}` | Stream de velas en tiempo real |

Documentación interactiva: `http://localhost:8000/docs`

---

## Parámetros de configuración

| Parámetro | Valor | Archivo |
|-----------|-------|---------|
| `MIN_PROBABILITY` | 78% | `trader.py` |
| `HISTORY_COUNT` | 300 velas | `trader.py` |
| `streak_percentile_min` | 70-80% | `strategy_config.yaml` |
| `half_life_max_candles` | 20-30 | `strategy_config.yaml` |
| `rel_atr_min / max` | 0.40 / 2.50 | `strategy_config.yaml` |
| `MIN_NEW_TRADES` (retrain) | 50 | `retrain_scheduler.py` |
| `CHECK_INTERVAL_H` | 6h | `retrain_scheduler.py` |
| `MAX_VERSIONS` | 5 | `retrain_scheduler.py` |
| Criterio AUC mínimo | 0.60 | `train_model.py` |
| Criterio Brier máximo | 0.24 | `train_model.py` |

---

## Estructura del proyecto

```
analisis mercados/
├── main.py                      # Servidor FastAPI + endpoints
├── trader.py                    # Orquestador: bucle, gatillos, motor ML/LLM
├── paper_trader.py              # Forward test: simula órdenes, registra en DB
├── indicators.py                # Build DataFrame + indicadores OTC
├── regime_filter.py             # 8 filtros de régimen OTC
├── generator_drift_detector.py  # Detector de cambios en el PRNG del broker
├── ml_classifier.py             # MLClassifier: LightGBM + Platt scaling
├── train_model.py               # Entrenamiento CLI: walk-forward + métricas
├── retrain_scheduler.py         # Loop de reentrenamiento automático
├── backtester.py                # Backtest vela a vela (sin look-ahead)
├── database.py                  # SQLite trades.db: schema + migrations
├── iqservice.py                 # Wrapper Exnova API
├── ai_brain.py                  # LLM Ollama (fallback cuando no hay modelo ML)
├── strategy_config.yaml         # Perfiles OTC por activo y franja horaria
├── models/
│   ├── lgbm_model.pkl           # Modelo champion activo
│   ├── platt_calibrator.pkl     # Calibrador Platt
│   ├── retrain_state.json       # Estado del scheduler
│   └── versions/                # Historial de modelos entrenados
├── trades.db                    # SQLite: historial completo de trades
├── tests/
│   ├── test_database.py         # 21 tests
│   ├── test_backtester.py       # 28 tests
│   ├── test_regime_filter.py    # 43 tests
│   ├── test_drift_detector.py   # 34 tests
│   ├── test_indicators_otc.py   # 43 tests
│   ├── test_ml_classifier.py    # 45 tests
│   └── test_retrain_scheduler.py # 27 tests
├── BROKER_INTEGRATION.md        # Integración con Exnova
├── AUDIT.md                     # Auditoría inicial del código
└── README.md                    # Este archivo
```

---

## Tests

```bash
# Suite completa (241 tests)
.venv/bin/pytest tests/ -v

# Por módulo
.venv/bin/pytest tests/test_ml_classifier.py -v
.venv/bin/pytest tests/test_retrain_scheduler.py -v

# Con cobertura
.venv/bin/pytest tests/ --cov=. --cov-report=term-missing
```

---

## Runbook operacional

### Arranque normal
```bash
source .venv/bin/activate
python main.py --mode paper &
sleep 3
curl -X POST http://localhost:8000/connect -d "email=$EMAIL&password=$PASS"
curl -X POST http://localhost:8000/bot/start \
  -H "Content-Type: application/json" \
  -d '{"asset": "EURUSD-OTC", "asset_type": "binary", "amount": 1.0}'
```

### El bot se apaga automáticamente (AUTOSHUTDOWN)
Ocurre por: 3 pérdidas diarias, 3 pérdidas consecutivas, drift detectado, o payout bajo.
```bash
# Ver razón del apagado
curl http://localhost:8000/bot/status | python3 -m json.tool | grep -A2 "last_decision"

# Reiniciar después de analizar
curl -X POST http://localhost:8000/bot/start \
  -H "Content-Type: application/json" \
  -d '{"asset": "EURUSD-OTC", "asset_type": "binary", "amount": 1.0}'
```

### Primer entrenamiento ML
```bash
# Verificar cantidad de datos
python3 -c "
from database import fetch_training_data
rows = fetch_training_data()
wins = sum(1 for r in rows if r['result']=='WIN')
print(f'Total: {len(rows)} | Wins: {wins} | WR: {wins/len(rows):.1%}' if rows else 'Sin datos')
"

# Entrenar (requiere ≥ 50 trades WIN/LOSS)
python train_model.py

# Verificar modelo cargado
curl http://localhost:8000/bot/status | python3 -m json.tool | grep model
```

### Drift detectado (el broker cambió su algoritmo)
```bash
# El bot se habrá apagado. Ver estado del detector:
python3 -c "
from generator_drift_detector import DriftDetector
d = DriftDetector.load()
print(d._state)
"
# Opciones:
# 1. Esperar 7 días (drift_state.json expira)
# 2. Limpiar manualmente y recolectar nuevos datos:
python3 -c "
from generator_drift_detector import DriftDetector
d = DriftDetector.load()
d.clear_drift('EURUSD-OTC')
d.save()
print('Drift limpiado para EURUSD-OTC')
"
# 3. Reentrenar el modelo con datos recientes
python retrain_scheduler.py --force
```

### Cambiar de activo
```bash
curl -X POST http://localhost:8000/bot/stop
curl -X POST http://localhost:8000/bot/start \
  -H "Content-Type: application/json" \
  -d '{"asset": "GBPJPY-OTC", "asset_type": "binary", "amount": 1.0}'
```

### Rollback de modelo
```bash
# Listar versiones disponibles
ls models/versions/

# Restaurar versión anterior (reemplazar champion)
cp models/versions/YYYYMMDD_HHMMSS/lgbm_model.pkl models/lgbm_model.pkl
cp models/versions/YYYYMMDD_HHMMSS/platt_calibrator.pkl models/platt_calibrator.pkl

# Reiniciar servidor para recargar modelo
curl -X POST http://localhost:8000/shutdown
python main.py --mode paper &
```

---

## Salvaguardas automáticas

| Evento | Acción |
|--------|--------|
| 3 pérdidas en el día | Apagado automático (AUTOSHUTDOWN) |
| 3 pérdidas consecutivas | Apagado automático (AUTOSHUTDOWN) |
| Payout < 80% | Bloqueo de ciclo (REGIME_BLOCK) |
| ATR relativo fuera de [0.40, 2.50] | Bloqueo de ciclo (REGIME_BLOCK) |
| Drift estadístico detectado | Apagado automático (AUTOSHUTDOWN) |
| Proba ML < 78% | No operar (GATE-1) |
| Dirección incoherente con racha | No operar (GATE-2) |
| ≥ 5 trades en el día | Bloqueo por límite diario |

---

## Aviso legal

Este software es para fines de investigación y educación. El trading de opciones
binarias conlleva riesgo significativo de pérdida de capital. El rendimiento pasado
**no garantiza resultados futuros**, especialmente en mercados OTC donde el edge
estadístico puede desaparecer sin previo aviso. Operar siempre en cuenta PRACTICE
antes de arriesgar capital real. El autor no se hace responsable de pérdidas derivadas
del uso de este software.
