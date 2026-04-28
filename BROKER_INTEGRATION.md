# Integración con Exnova — Broker Integration Guide

Documentación técnica de cómo el bot se conecta a Exnova, qué endpoints usa,
y cuáles son los puntos de fragilidad conocidos.

---

## ¿Qué es Exnova?

Exnova es un broker de opciones binarias que fue rebrandeado desde IQ Option.
Comparte la misma infraestructura de API y WebSocket que IQ Option, con la URL
de conexión cambiada de `iqoption.com` a `trade.exnova.com`.

El mercado OTC de Exnova genera precios **internamente** mediante un algoritmo
privado (PRNG). No hay feed externo de mercado real. Los precios OTC son únicos
por broker y no replicables en ningún exchange.

---

## Librería base

Se usa `iqoptionapi` (PyPI) con monkey-patching en `iqservice.py` para redirigir
las URLs al dominio de Exnova:

```python
# iqservice.py — redirección de URLs
import iqoptionapi.constants as _c
_c.TRADING_HOSTNAME  = "trade.exnova.com"
_c.SSE_URL           = "https://trade.exnova.com/api/..."
```

La librería maneja:
- Autenticación via HTTP (email/password → sesión con cookie)
- WebSocket persistente para streaming de velas y resultados de órdenes
- Colocación de órdenes binary/turbo/digital

---

## Flujo de conexión

```
1. IQOptionAPI(email, password)
2. api.connect()  → POST https://trade.exnova.com/login
                    respuesta: 200 OK + cookies de sesión
3. api.change_balance("PRACTICE")  → cuenta demo (recomendado)
4. WebSocket abierto automáticamente en ws://trade.exnova.com/echo/websocket
```

El bot verifica la conexión en cada ciclo antes de operar:
```python
iq_service.is_connected()  # → bool
```

---

## Endpoints y operaciones usadas

### Obtener velas históricas
```python
# Interno: WebSocket subscribeMessage con candles
api.get_candles(asset, interval_seconds, count, end_time)
# Retorna: lista de dicts {id, from, to, open, close, high, low, volume}
```

### Streaming de velas en tiempo real
```python
api.start_candles_stream(asset, interval_seconds)
api.stop_candles_stream(asset, interval_seconds)
# Actualiza un buffer interno; se lee con get_realtime_candles()
```

### Colocar orden binaria
```python
api.buy(amount, asset, direction, expiration_time)
# direction: "call" o "put"
# expiration_time: 1, 2, 3, 4, 5 (minutos)
# Retorna: (success: bool, order_id: int)
```

### Colocar orden digital
```python
api.buy_digital_spot(asset, amount, direction, duration)
# Retorna: (success: bool, order_id: int)
```

### Consultar resultado de orden
```python
# Binary: WebSocket event "option-closed"
# Digital: WebSocket event "position-changed" con status="closed"
api.get_async_order(order_id)
```

### Obtener payout actual
```python
api.get_all_profit()
# Retorna: dict {asset: {"binary": payout_float, "turbo": payout_float}}
# Ejemplo: {"EURUSD-OTC": {"binary": 0.82}}
```

---

## Activos OTC disponibles

Los activos OTC tienen el sufijo `-OTC`. Algunos de los más comunes:

| Activo | Tipo | Disponibilidad |
|--------|------|----------------|
| `EURUSD-OTC` | Forex OTC | 24/7 |
| `GBPJPY-OTC` | Forex OTC | 24/7 |
| `USDJPY-OTC` | Forex OTC | 24/7 |
| `AUDCAD-OTC` | Forex OTC | 24/7 |
| `EURGBP-OTC` | Forex OTC | 24/7 |

Los activos OTC siempre están disponibles (24/7, incluyendo fines de semana),
a diferencia de los activos de mercado real que tienen horario de sesión.

---

## Puntos de fragilidad conocidos

### 1. Reconexión WebSocket
La conexión WebSocket se corta frecuentemente (timeout, cambios de red).
La librería `iqoptionapi` tiene lógica de reconexión automática, pero puede
fallar silenciosamente. El bot detecta esto si `get_candles` devuelve lista vacía.

**Mitigación:** `iq_service.is_connected()` verifica el estado antes de cada ciclo.
Si falla, el ciclo hace skip y espera la próxima vela.

### 2. Cambios de API sin previo aviso
Exnova puede cambiar endpoints, formato de respuesta o comportamiento de la API
sin anuncio previo. La librería `iqoptionapi` no tiene soporte oficial.

**Mitigación:** El bot usa fail-open en todas las consultas externas (si falla,
continúa con valor por defecto). Monitorear los logs cuando el bot falla repetidamente.

### 3. Payout variable
El payout no es constante. Puede bajar al 70% o menos en determinadas horas.
El `payout_filter` en `regime_filter.py` bloquea operaciones cuando el payout
cae bajo el 80% (configurable).

**Síntoma:** REGIME_BLOCK frecuentes con filtro `payout_filter` → el broker está
ofreciendo payout bajo. Reducir el umbral o pausar manualmente.

### 4. Rate limiting y ban de IP
Consultas muy frecuentes a la API pueden resultar en ban temporal de IP.
El bot hace una consulta de velas por ciclo (60s), lo que está dentro de los
límites observados. No hacer múltiples instancias concurrentes.

### 5. Cambio de algoritmo OTC (el riesgo principal)
Exnova puede modificar su PRNG en cualquier momento. Cuando esto ocurre:
- El modelo ML entrenado pierde su edge estadístico
- El `generator_drift_detector.py` lo detecta en 1-7 días según la magnitud
- El bot se apaga automáticamente al detectar drift significativo (2+ de 4 tests estadísticos)

**Señales de alerta:** win rate cae abruptamente por debajo del 50%, el detector
de drift muestra KS > 0.10 o |Z| > 2.58 en múltiples activos simultáneamente.

### 6. Slippage en órdenes
Entre el momento en que el bot decide operar y el momento en que la orden se
ejecuta pueden pasar 0.5-2 segundos (latencia de red + procesamiento del broker).
En velas M1, 2 segundos es ~3% del tiempo total de la vela, lo que puede
desplazar el precio de entrada significativamente.

**Mitigación:** Operar siempre en los primeros 30s de la vela (el bot lo hace
al detectar el cierre de vela, no al inicio).

---

## Autenticación y credenciales

Las credenciales se leen de variables de entorno via `python-dotenv`:

```bash
# .env (no commitear al repositorio)
EXNOVA_EMAIL=tu@email.com
EXNOVA_PASSWORD=tu_password
```

```python
# iqservice.py
load_dotenv()
email    = os.getenv("EXNOVA_EMAIL")
password = os.getenv("EXNOVA_PASSWORD")
```

El archivo `.env` está en `.gitignore`. Nunca commitear credenciales.

---

## Debugging de conexión

```bash
# Test básico de conexión (sin credenciales hardcodeadas en código)
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
from iqservice import iq_service
ok = iq_service.connect(os.getenv('EXNOVA_EMAIL'), os.getenv('EXNOVA_PASSWORD'))
print('Conectado:', ok)
print('Balance:', iq_service.api.get_balance() if ok else 'N/A')
"

# Ver payouts actuales
python3 -c "
from iqservice import iq_service
# (después de conectar)
payouts = iq_service.api.get_all_profit()
for asset, data in sorted(payouts.items()):
    if 'OTC' in asset:
        print(f'{asset}: {data}')
"
```

---

## Consideraciones de seguridad

1. **No commitear `.env`** — está en `.gitignore`
2. **Usar cuenta PRACTICE** hasta validar el sistema (mínimo 30 días)
3. **No exponer el puerto 8000** a internet — el servidor es local
4. **El archivo `test_conexion.py`** contiene credenciales hardcodeadas (legacy) — no modificar ni commitear
5. **`trades.db`** contiene historial de operaciones — hacer backup periódico pero no subir al repo (puede contener información sensible de timing)
