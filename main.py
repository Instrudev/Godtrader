"""
main.py – Servidor FastAPI del Bot de Trading con Ollama v2
API REST para control del bot + WebSocket de velas en tiempo real.

Endpoints principales:
  POST /connect           → Autenticar con IQ Option
  GET  /assets            → Listar activos disponibles
  GET  /candles/{asset}   → Histórico de 300 velas
  POST /bot/start         → Iniciar el bot de trading autónomo
  POST /bot/stop          → Detener el bot
  GET  /bot/status        → Estado completo + estadísticas + log de operaciones
  WS   /ws/{asset}        → Stream de velas en tiempo real

Modos de operación (flag --mode):
  live    (default) → opera con dinero real en la cuenta activa del broker
  paper             → simula órdenes usando precios reales; registra en trades.db
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Selección de modo (live / paper) ────────────────────────────────────────
# Se parsea antes de cargar FastAPI para que el singleton del bot sea el correcto.

def _parse_mode() -> str:
    """Lee --mode de argv sin interferir con uvicorn cuando se importa como módulo."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["live", "paper"], default="live")
    args, _ = parser.parse_known_args()
    return args.mode


_BOT_MODE = _parse_mode()

from asset_scanner import asset_scanner
from iqservice import iq_service
from retrain_scheduler import retrain_scheduler

if _BOT_MODE == "paper":
    from paper_trader import paper_trading_bot as trading_bot
    logging.getLogger(__name__).info("═══ MODO PAPER TRADING ═══ (sin órdenes reales)")
else:
    from trader import trading_bot

# ─── Configuración de Logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from iqservice import REMEDIATION_MODE, FORCE_DEMO_ACCOUNT
    if REMEDIATION_MODE:
        logger.warning(
            "═══ REMEDIATION MODE — DEMO ONLY ═══ "
            f"(FORCE_DEMO_ACCOUNT={FORCE_DEMO_ACCOUNT})"
        )
        logger.warning("REMEDIATION MODE — retrain_scheduler disabled")
    else:
        await retrain_scheduler.start()
    yield
    # Shutdown
    logger.info("Deteniendo el servidor y garantizando la desocupación del puerto...")
    retrain_scheduler.stop()
    if trading_bot.running:
        trading_bot.stop()
    if asset_scanner.running:
        asset_scanner.stop()

    if hasattr(iq_service, 'api') and hasattr(iq_service.api, 'close'):
        try:
            iq_service.api.close()
        except Exception:
            pass

    await asyncio.sleep(0.5)
    # Fuerza el cierre de todos los hilos rebeldes para liberar el puerto
    os._exit(0)

# ─── Aplicación FastAPI ───────────────────────────────────────────────────────

app = FastAPI(
    title="IQ Option AI Trading Bot – Ollama Edition",
    description=(
        "Bot de trading autónomo con análisis técnico local (numpy/pandas) "
        "y decisión IA por phi3 via Ollama. "
        "Optimizado para Apple Silicon M5."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Archivos estáticos (HTML/CSS/JS del frontend – opcional)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    def read_root():
        return FileResponse("static/index.html")

except RuntimeError:
    # Directorio static no encontrado; ignorar silenciosamente
    @app.get("/", include_in_schema=False)
    def read_root():
        return {"status": "IQ Option AI Bot v2 – Ollama Edition", "docs": "/docs"}


# ─── Autenticación ────────────────────────────────────────────────────────────

@app.post("/connect", summary="Conectar a IQ Option")
def connect(email: str = Form(...), password: str = Form(...)):
    """Establece conexión con IQ Option usando las credenciales del usuario."""
    check, reason = iq_service.connect(email, password)
    if not check:
        raise HTTPException(status_code=401, detail=f"Conexión fallida: {reason}")
    return {"status": "ok", "message": "Conexión exitosa", "email": email}


# ─── Datos de Mercado ─────────────────────────────────────────────────────────

@app.get("/assets", summary="Listar activos disponibles")
def get_assets():
    """Retorna todos los activos con su estado abierto/cerrado."""
    if not iq_service.is_connected():
        raise HTTPException(status_code=403, detail="No conectado a IQ Option")
    return {"assets": iq_service.get_assets()}


@app.get("/candles/{asset_id}", summary="Histórico de velas")
def get_candles(asset_id: str, interval: int = 60, count: int = 300):
    """
    Retorna el histórico de velas del activo. Por defecto: 300 velas de 60s.
    Las velas se guardan automáticamente en candles_history para backtesting.
    """
    if not iq_service.is_connected():
        raise HTTPException(status_code=403, detail="No conectado a IQ Option")
    try:
        from database import init_db, save_candles
        candles = iq_service.get_candles(asset_id, interval, count)
        if candles:
            init_db()
            saved = save_candles(asset_id, candles, interval_s=interval)
            logger.info(f"Velas guardadas en BD: {saved} velas de {asset_id} ({interval}s)")
        return {"candles": candles, "count": len(candles), "saved_to_db": len(candles) > 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Control del Bot ──────────────────────────────────────────────────────────

class BotStartRequest(BaseModel):
    asset: str = Field(
        ...,
        json_schema_extra={"example": "EURUSD"},
        description="Nombre del activo a operar",
    )
    asset_type: str = Field(
        "binary",
        json_schema_extra={"example": "binary"},
        description="Tipo de opción: 'binary', 'turbo' o 'digital'",
    )
    amount: float = Field(
        1000.0,
        gt=0,
        json_schema_extra={"example": 1.0},
        description="Monto por operación en la moneda de la cuenta",
    )


@app.post("/bot/start", summary="Iniciar el bot de trading")
async def bot_start(req: BotStartRequest):
    """
    Inicia el bot de trading autónomo.

    Flujo por vela:
    1. Filtro de Pre-Calificación (RSI extremo + toque BB) → si falla, skip silencioso
    2. Análisis IA con phi3 local (Ollama en M5 via Metal GPU)
    3. Doble gatillo: pr ≥ 85% AND dirección == tendencia EMA200
    4. Ejecución con api.buy() si ambos gatillos superados
    """
    if not iq_service.is_connected():
        raise HTTPException(status_code=403, detail="No conectado a IQ Option")
    if trading_bot.running:
        raise HTTPException(status_code=409, detail="El bot ya está en ejecución")

    ok = trading_bot.start(req.asset, req.asset_type, req.amount)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo iniciar el bot")

    return {
        "status": "ok",
        "message": f"Bot iniciado en {req.asset}",
        "mode": _BOT_MODE,
        "config": req.model_dump(),
        "thresholds": {
            "min_probability": 78,
            "pre_qual": "RSI<35 o RSI>65 + precio en zona exterior BB",
            "ema_filter": "CALL solo si EMA20 o EMA200 bullish / PUT solo si EMA20 o EMA200 bearish",
        },
    }


@app.post("/bot/stop", summary="Detener el bot")
async def bot_stop():
    """Envía señal de parada. El ciclo actual finalizará antes de detenerse."""
    if not trading_bot.running:
        raise HTTPException(status_code=400, detail="El bot no está en ejecución")
    trading_bot.stop()
    return {"status": "ok", "message": "Señal de parada enviada"}


@app.post("/shutdown", summary="Apagar servidor y liberar puerto")
def shutdown_server():
    """Cierra completamente la sesión y apaga el proceso para liberar el puerto 8000."""
    logger.warning("Recibida petición de apagado total desde el frontend.")
    if trading_bot.running:
        trading_bot.stop()
    if hasattr(iq_service, 'api') and hasattr(iq_service.api, 'close'):
        try:
            iq_service.api.close()
        except:
            pass
            
    async def kill_proc():
        await asyncio.sleep(0.5)
        os._exit(0)  # Cierre radical de la aplicación
        
    asyncio.create_task(kill_proc())
    return {"status": "ok", "message": "El servidor se apagará en un segundo."}


@app.get("/bot/status", summary="Estado del bot y estadísticas")
def bot_status():
    """
    Estado completo del bot:
    - Configuración y modelo IA
    - Última decisión del motor (ML o LLM fallback)
    - Estadísticas de ciclos, pre-calificaciones y operaciones
    - Log de las últimas 20 operaciones
    - Win rate calculado
    """
    return trading_bot.get_status()


# ─── Scanner Multi-Activo ────────────────────────────────────────────────────

# Activos OTC más comunes en Exnova como lista por defecto
_DEFAULT_OTC_ASSETS = [
    "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDUSD-OTC",
    "EURGBP-OTC", "EURJPY-OTC", "GBPJPY-OTC", "USDCHF-OTC",
]


class ScannerStartRequest(BaseModel):
    assets: List[str] = Field(
        default_factory=list,
        description="Activos a escanear. Vacío = auto-detecta todos los binarios/turbo abiertos.",
    )
    asset_type: str = Field("binary", description="Tipo: 'binary', 'turbo' o 'digital'")
    amount: float   = Field(1.0, gt=0, description="Monto por operación")
    expiry_min: int = Field(2, ge=1, le=5, description="Minutos de expiración")
    mode: str       = Field(
        "practice",
        description="'practice' = ejecuta órdenes reales en cuenta demo | 'paper' = simulado sin API",
    )


@app.post("/scanner/start", summary="Iniciar scanner multi-activo")
async def scanner_start(req: ScannerStartRequest):
    """
    Inicia el scanner multi-activo.

    Cada cierre de vela (60s) escanea todos los activos disponibles,
    identifica la señal más fuerte y ejecuta si el ML confirma.

    mode='practice' → llama a buy_binary sobre la cuenta PRÁCTICA del broker.
    mode='paper'    → simula localmente sin tocar la API del broker.
    """
    if not iq_service.is_connected():
        raise HTTPException(status_code=403, detail="No conectado al broker")
    if asset_scanner.running:
        raise HTTPException(status_code=409, detail="El scanner ya está en ejecución")

    # "practice" → modo "live" internamente (el broker ya está en PRÁCTICA)
    internal_mode = "paper" if req.mode == "paper" else "live"

    assets = req.assets if req.assets else []  # lista vacía = auto-detecta
    ok = asset_scanner.start(
        assets     = assets,
        asset_type = req.asset_type,
        amount     = req.amount,
        expiry_min = req.expiry_min,
        mode       = internal_mode,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo iniciar el scanner")

    label = "PRÁCTICA" if req.mode == "practice" else "SIMULADO (paper)"
    return {
        "status":  "ok",
        "message": f"Scanner iniciado — modo {label}",
        "mode":    req.mode,
        "amount":  req.amount,
    }


@app.post("/scanner/stop", summary="Detener scanner multi-activo")
async def scanner_stop():
    if not asset_scanner.running:
        raise HTTPException(status_code=400, detail="El scanner no está en ejecución")
    asset_scanner.stop()
    return {"status": "ok", "message": "Scanner detenido"}


@app.get("/scanner/status", summary="Estado del scanner multi-activo")
def scanner_status():
    """
    Estado completo del scanner:
    - Lista de activos monitoreados
    - Resultado del último scan (qué señales encontró en cada activo)
    - Stats: scans totales, señales, trades ejecutados, winrate
    - Log de los últimos 20 trades
    """
    return asset_scanner.get_status()


# ─── Backtesting ─────────────────────────────────────────────────────────────

@app.post("/backtest/{asset_id}", summary="Backtest sobre velas guardadas en BD")
def run_backtest(
    asset_id: str,
    interval: int = 60,
    limit: int = 500,
    payout: float = 0.80,
    amount: float = 1.0,
):
    """
    Ejecuta el backtester sobre las velas guardadas en candles_history.

    Primero llama a GET /candles/{asset_id} para asegurarte de tener datos.
    Devuelve el reporte completo con winrate, profit factor, drawdown y más.
    """
    from backtester import Backtester
    bt = Backtester(payout=payout, amount=amount, save_to_db=False)
    try:
        report = bt.run_from_db(asset_id, interval_s=interval, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "asset":            report.asset,
        "date_from":        report.date_from,
        "date_to":          report.date_to,
        "total_candles":    report.total_candles,
        "total_signals":    report.total_signals,
        "total_trades":     report.total_trades,
        "wins":             report.wins,
        "losses":           report.losses,
        "winrate":          round(report.winrate * 100, 1),
        "profit_factor":    round(report.profit_factor, 2),
        "net_profit":       round(report.net_profit, 2),
        "max_drawdown_pct": round(report.max_drawdown_pct, 1),
        "sharpe":           round(report.sharpe, 2),
        "monthly_returns":  report.monthly_returns,
        "hourly_winrate":   {str(k): round(v * 100, 1) for k, v in report.hourly_winrate.items()},
        "weekday_winrate":  {str(k): round(v * 100, 1) for k, v in report.weekday_winrate.items()},
        "summary":          report.summary(),
    }


# ─── Endpoints de reentrenamiento ML ─────────────────────────────────────────

@app.get("/retrain/status", summary="Estado del scheduler de reentrenamiento")
def retrain_status():
    """
    Estado del loop de reentrenamiento automático:
    - Si está corriendo, intervalo, condiciones
    - Último entreno y resultado
    - Historial de versiones
    """
    return retrain_scheduler.get_status()


@app.post("/retrain/trigger", summary="Forzar reentrenamiento inmediato")
async def retrain_trigger():
    """
    Dispara un ciclo de reentrenamiento ahora, ignorando el intervalo temporal
    pero respetando la condición de trades mínimos nuevos.
    Usar /retrain/force para ignorar también esa condición.
    """
    msg = retrain_scheduler.trigger_now()
    return {"status": "ok", "message": msg}


@app.get("/trades", summary="Historial de operaciones")
def get_trades(
    mode: str = None,
    asset: str = None,
    result: str = None,
    limit: int = 200,
):
    """
    Devuelve el historial de operaciones guardado en la BD.
    Parámetros opcionales: mode (live/paper/backtest), asset, result (WIN/LOSS/PENDING).
    """
    from database import fetch_trades
    trades = fetch_trades(mode=mode, asset=asset, limit=limit)
    if result:
        trades = [t for t in trades if t.get("result") == result.upper()]
    return {"trades": trades, "total": len(trades)}


@app.post("/retrain/force", summary="Forzar reentrenamiento sin condiciones")
async def retrain_force():
    """
    Reentrena ignorando tanto el intervalo como la condición de trades nuevos.
    Útil para el primer entrenamiento manual.
    """
    import asyncio as _asyncio
    _asyncio.create_task(
        _asyncio.to_thread(retrain_scheduler._run_retrain)
    )
    return {"status": "ok", "message": "Reentrenamiento forzado iniciado en background."}


# ─── WebSocket – Stream de Velas y Notificaciones en Tiempo Real ──────────────

active_notifications_ws = set()

async def broadcast_notification(event: dict):
    # Enviar a todos los websockets de notificaciones activos
    for ws in list(active_notifications_ws):
        try:
            await ws.send_json(event)
        except Exception:
            pass

# Conectar el trader y el scanner con nuestra función de broadcast
trading_bot.on_notification = broadcast_notification
asset_scanner.on_notification = broadcast_notification

@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    """
    WebSocket específico para "empujar" las notificaciones Push
    y cambios en el historial de señales generados por el bot.
    """
    await websocket.accept()
    active_notifications_ws.add(websocket)
    # Enviar un saludo y el historial guardado actual si existe
    await websocket.send_json({
        "type": "HISTORY",
        "message": "Connected",
        "data": {"history": trading_bot.signal_history}
    })
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_notifications_ws.remove(websocket)


@app.websocket("/ws/{asset_id}")
async def websocket_candles(
    websocket: WebSocket, asset_id: str, interval: int = 60
):
    """
    Stream WebSocket de velas en tiempo real.
    Envía la vela actual cada vez que el precio o el timestamp cambia (~0.5s).
    """
    if not iq_service.is_connected():
        await websocket.close(code=1008)
        return

    await websocket.accept()
    started = iq_service.start_stream(asset_id, interval)

    if not started:
        await websocket.send_text(json.dumps({"error": "No se pudo iniciar el stream"}))
        await websocket.close()
        return

    await asyncio.sleep(2)
    last_close = None
    last_time  = None

    try:
        while True:
            candle = iq_service.get_realtime_candle(asset_id, interval)
            if candle and (
                candle["close"] != last_close or candle["time"] != last_time
            ):
                await websocket.send_text(json.dumps(candle))
                last_close = candle["close"]
                last_time  = candle["time"]
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    finally:
        iq_service.stop_stream(asset_id, interval)


# ─── Punto de Entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Autocleanup de puertos zombies (Robusto para M5/Mac)
    import subprocess
    try:
        pid_lines = subprocess.check_output(["lsof", "-t", "-i:8000"], text=True).strip().split('\n')
        for pid in pid_lines:
            if pid:
                subprocess.run(["kill", "-9", pid])
                print(f"Puerto 8000 liberado a la fuerza (Proceso Zombie {pid} eliminado).")
    except Exception:
        pass  # No había ningún proceso ocupándolo

    # reload=False es OBLIGATORIO: reload destruye el event loop y el estado del bot
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
