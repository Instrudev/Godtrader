"""
asset_scanner.py – Escáner multi-activo para el bot de trading

Escanea una lista de activos cada cierre de vela (60s), identifica la señal
más fuerte entre todos usando pre_qualify() + scoring, y ejecuta el trade
en el activo ganador si el ML confirma con proba ≥ umbral.

Diferencia clave con trader.py:
  trader.py      → bucle infinito en UN activo fijo
  asset_scanner  → cada vela, elige el MEJOR activo de la lista y opera ahí

Flujo por ciclo:
  1. Esperar cierre de vela (alineado a 60s UTC)
  2. Descargar velas de cada activo (con delay 0.35s entre requests)
  3. Ejecutar pre_qualify() en cada uno
  4. Puntuar señales → elegir la más fuerte
  5. ML valida la dirección de la señal elegida (proba ≥ 78%)
  6. Si supera: ejecutar trade + cooldown global
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from database import (
    DB_PATH, init_db, insert_trade, get_winrate_by_hour,
    update_trade_result_simple, update_trade_post_analysis,
)
from indicators import (
    build_dataframe,
    compute_prng_features,
    detect_bb_body_reversal,
    detect_bb_body_reversal_call,
    detect_bb_two_candle_reversal,
    get_streak_info,
    pre_qualify,
)
from iqservice import (
    iq_service,
    ML_DISABLED_MODE,
    ML_DISABLED_MIN_SCORE,
    ML_DISABLED_MAX_ASSET_LOSSES,
    ML_DISABLED_MAX_DAILY_LOSSES,
    ML_DISABLED_MAX_DAILY_TRADES,
)
from ml_classifier import ml_classifier, extract_features
from regime_filter import (
    check_all_filters,
    MAX_ASSET_DAILY_LOSSES,
    MAX_DAILY_LOSSES,
    MAX_TRADES_PER_DAY,
)

logger = logging.getLogger(__name__)

# ─── Logger JSONL para decisiones de estrategia ──────────────────────────────

_decision_logger = logging.getLogger("strategy_decisions")
_decision_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_decision_log_dir, exist_ok=True)
_decision_handler = logging.FileHandler(
    os.path.join(_decision_log_dir, "strategy_decisions.log"), encoding="utf-8"
)
_decision_handler.setFormatter(logging.Formatter("%(message)s"))
_decision_logger.addHandler(_decision_handler)
_decision_logger.setLevel(logging.INFO)


# ─── Señal de estrategia ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategySignal:
    """Resultado de evaluación de una estrategia individual."""
    active: bool
    direction: Optional[str]   # "CALL" | "PUT" | None
    score: float
    strategy_name: str


# ─── Constantes ───────────────────────────────────────────────────────────────

CANDLE_INTERVAL     = 60    # segundos por vela
HISTORY_COUNT       = 300   # velas a solicitar por activo
MIN_PROBABILITY     = 55    # umbral ML — activo solo cuando ML_MIN_TRADES está cubierto
ML_MIN_TRADES       = 150   # trades cerrados mínimos para que el ML sea confiable
REQUEST_DELAY       = 0.5   # delay entre requests al broker (reducido para acelerar scan)
SCAN_EXTRA_SECS     = 1     # segundos extra tras cierre para que el dato esté disponible
MAX_CONSEC_ERRORS   = 2     # errores consecutivos antes de abortar el scan
GET_CANDLES_TIMEOUT = 7.0   # segundos máx por llamada — evita bloqueo indefinido del WS
MAX_ASSETS_PER_SCAN = 20    # máximo de activos escaneados por ciclo (rota aleatoriamente)
MAX_ENTRY_SECS      = 15    # segundos máx desde inicio de vela para entrar — evita entrar a destiempo
PRE_SCAN_SECS       = 15    # segundos ANTES del cierre de vela para iniciar pre-scan
MAX_HOT_CANDIDATES  = 5     # candidatos calientes a re-verificar tras cierre


# ─── Clase Principal ──────────────────────────────────────────────────────────

class AssetScanner:
    """
    Escáner multi-activo.

    Cada cierre de vela:
      1. Descarga velas de cada activo en la lista
      2. Evalúa pre_qualify() en todos
      3. Puntúa las señales y elige la más fuerte
      4. ML valida la dirección → si proba ≥ umbral, ejecuta
      5. Cooldown global hasta que expire el trade
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.assets: List[str] = []
        self.asset_type: str = "binary"
        self.amount: float = 1.0
        self.expiry_min: int = 2
        self.mode: str = "paper"
        self._task: Optional[asyncio.Task] = None
        self._cooldown_until: float = 0.0
        self.on_notification = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self.total_scans: int = 0
        self.signals_found: int = 0
        self._strategy_telemetry: Dict[str, int] = self._fresh_telemetry()
        self.trades_executed: int = 0
        self.trade_log: List[Dict] = []
        self.last_scan_result: List[Dict] = []
        self.last_scan_time: Optional[str] = None

    # ─── Interfaz pública ─────────────────────────────────────────────────────

    def start(
        self,
        assets: List[str],
        asset_type: str = "binary",
        amount: float = 1.0,
        expiry_min: int = 2,
        mode: str = "paper",
    ) -> bool:
        if self.running:
            logger.warning("[SCANNER] Ya está en ejecución.")
            return False
        if not iq_service.is_connected():
            logger.error("[SCANNER] Sin conexión al broker.")
            return False

        self.loop = asyncio.get_running_loop()

        # Lista vacía = auto-detectar todos los activos binarios/turbo abiertos cada ciclo
        self.assets          = list(assets)
        self.asset_type      = asset_type
        self.amount          = amount
        self.expiry_min      = expiry_min
        self.mode            = mode
        self.running         = True
        self._cooldown_until = 0.0
        self.total_scans     = 0
        self.signals_found   = 0
        self.trades_executed = 0
        self.last_scan_result = []
        self.last_scan_time  = None

        if not ml_classifier.is_loaded():
            ml_classifier.load()

        init_db(DB_PATH)
        self.trade_log = self._reconstruct_trade_log()
        self._task = asyncio.create_task(self._scan_loop())

        logger.info("=" * 60)
        if self.assets:
            logger.info(
                f"[SCANNER] INICIADO | {len(self.assets)} activos fijos | "
                f"{asset_type.upper()} | ${amount:.2f}/op | modo={mode}"
            )
        else:
            logger.info(
                f"[SCANNER] INICIADO | auto-detección de activos | "
                f"{asset_type.upper()} | ${amount:.2f}/op | modo={mode}"
            )
        logger.info("=" * 60)
        return True

    def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info(
            f"[SCANNER] DETENIDO | Scans={self.total_scans} | "
            f"Señales={self.signals_found} | Trades={self.trades_executed}"
        )

    def _reconstruct_trade_log(self) -> List[Dict]:
        """
        Reconstruye trade_log desde la BD con trades del día UTC actual.

        Mitiga la vulnerabilidad post-restart: si el bot se reinicia tras
        acumular pérdidas, los contadores de daily_loss y per_asset_loss
        se reconstruyen correctamente desde la fuente de verdad (BD).
        """
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT timestamp, asset, direction, result
                    FROM trades
                    WHERE timestamp >= ? AND timestamp < ?
                      AND result IN ('WIN', 'LOSS', 'TIE')
                    ORDER BY timestamp ASC
                    """,
                    (today, tomorrow),
                ).fetchall()

            records = [
                {
                    "timestamp": row["timestamp"],
                    "asset":     row["asset"],
                    "direction": row["direction"],
                    "result":    row["result"],
                }
                for row in rows
            ]
            logger.info(
                f"[SCANNER] Reconstruyendo trade_log desde BD: "
                f"{len(records)} trades del día UTC actual ({today}) cargados."
            )
            return records

        except Exception as e:
            logger.error(
                f"[SCANNER] Could not reconstruct trade_log: {e}. "
                "Daily limits may be incorrect."
            )
            # Log de seguridad para auditoría
            try:
                from iqservice import _security_logger
                ts = datetime.now(timezone.utc).isoformat()
                _security_logger.warning(
                    f"[{ts}] HALT_TYPE=TRADE_LOG_RECONSTRUCTION_FAILED "
                    f"exception={type(e).__name__}: {e} "
                    f"impact=Daily limits may be incorrect until next successful startup "
                    f"action_required=Manual verification of daily loss counters"
                )
            except Exception:
                pass
            return []

    def get_status(self) -> Dict:
        wins  = sum(1 for t in self.trade_log if t.get("result") == "WIN")
        total = len(self.trade_log)
        return {
            "running":            self.running,
            "assets":             self.assets,
            "asset_type":         self.asset_type,
            "amount":             self.amount,
            "mode":               self.mode,
            "cooldown_remaining": round(max(0.0, self._cooldown_until - time.time()), 1),
            "last_scan_time":     self.last_scan_time,
            "last_scan_result":   self.last_scan_result,
            "stats": {
                "total_scans":     self.total_scans,
                "signals_found":   self.signals_found,
                "trades_executed": self.trades_executed,
                "wins":            wins,
                "losses":          total - wins,
                "win_rate":        f"{wins/total*100:.1f}%" if total else "N/A",
            },
            "trade_log": self.trade_log[-20:],
        }

    # ─── Telemetría de estrategias ───────────────────────────────────────────

    @staticmethod
    def _fresh_telemetry() -> Dict[str, int]:
        from regime_filter import _today_utc
        return {
            "single_match": 0, "multiple_match": 0,
            "conflict_cancel": 0, "no_signal": 0,
            "session_date_utc": _today_utc(),
        }

    def _check_telemetry_reset(self) -> None:
        """Resetea contadores si el día UTC cambió."""
        from regime_filter import _today_utc
        today = _today_utc()
        if self._strategy_telemetry.get("session_date_utc") != today:
            logger.info(f"[TELEMETRY] Nuevo día UTC ({today}). Reseteando contadores.")
            self._strategy_telemetry = self._fresh_telemetry()

    def get_strategy_telemetry(self) -> Dict:
        """Snapshot de telemetría de estrategias del día actual."""
        self._check_telemetry_reset()
        return dict(self._strategy_telemetry)

    # ─── Bucle principal ──────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        logger.info("[SCANNER] Esperando primer cierre de vela...")

        while self.running:
            try:
                # ── FASE 1: PRE-SCAN (segundo ~45 de la vela actual) ─────────
                # Hace el trabajo pesado ANTES del cierre: fetch 300 velas,
                # build_dataframe, pre_qualify, score. Identifica candidatos
                # "calientes" que podrían dar señal en la próxima vela.
                now        = time.time()
                next_close = (now // CANDLE_INTERVAL + 1) * CANDLE_INTERVAL
                pre_scan_at = next_close - PRE_SCAN_SECS
                wait_pre = pre_scan_at - time.time()
                if wait_pre > 0:
                    await asyncio.sleep(wait_pre)

                if not self.running:
                    break

                # Cooldown activo → saltear ciclo completo
                remaining = self._cooldown_until - time.time()
                if remaining > 0:
                    logger.info(f"[SCANNER] Cooldown activo ({remaining:.0f}s restantes). Skip.")
                    await asyncio.sleep(min(remaining + 1, CANDLE_INTERVAL))
                    continue

                self.total_scans += 1
                ts_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
                logger.info(f"[SCANNER] ── Pre-scan #{self.total_scans} @ {ts_str} UTC {'─'*30}")

                # Pre-scan pesado: todos los activos en paralelo
                pre_candidates = await asyncio.to_thread(self._scan_all_assets)

                # Identificar candidatos calientes (califican o están cerca)
                # Ordenar por score para priorizar re-verificación
                hot = sorted(
                    [c for c in pre_candidates if c.get("df") is not None],
                    key=lambda x: x["score"],
                    reverse=True,
                )[:MAX_HOT_CANDIDATES]
                hot_assets = [c["asset"] for c in hot]

                pre_qualified = [c for c in pre_candidates if c["qualifies"]]
                if hot_assets:
                    logger.info(
                        f"[SCANNER] Pre-scan: {len(pre_qualified)} señal(es) | "
                        f"Hot: {', '.join(hot_assets)}"
                    )
                else:
                    logger.info("[SCANNER] Pre-scan: sin candidatos calientes.")

                # ── FASE 2: CONFIRMACIÓN (segundo ~2 de la vela nueva) ───────
                # Esperar al cierre real de la vela
                wait_close = next_close - time.time() + SCAN_EXTRA_SECS
                if wait_close > 0:
                    await asyncio.sleep(wait_close)

                if not self.running:
                    break

                # Re-verificar SOLO los candidatos calientes con datos frescos
                if not hot:
                    logger.info("[SCANNER] Sin candidatos para confirmar. Siguiente vela.")
                    self._notify("SCANNER_SCAN", "Sin señales", data={"scan": []})
                    continue

                logger.info(f"[SCANNER] Confirmando {len(hot)} candidato(s) con vela fresca...")
                candidates = await asyncio.to_thread(self._confirm_hot, hot)

                self.last_scan_time   = datetime.now(timezone.utc).isoformat()
                self.last_scan_result = [
                    {
                        "asset":     c["asset"],
                        "qualifies": c["qualifies"],
                        "direction": c["direction"],
                        "score":     c["score"],
                        "reason":    c["reason"][:80],
                    }
                    for c in candidates
                ]

                # Filtrar calificados y ordenar por score desc
                qualified = [c for c in candidates if c["qualifies"] and c["df"] is not None]
                qualified.sort(key=lambda x: x["score"], reverse=True)

                if not qualified:
                    logger.info("[SCANNER] Candidatos no confirmados tras cierre. Siguiente vela.")
                    self._notify("SCANNER_SCAN", "Sin señales confirmadas", data={"scan": self.last_scan_result})
                    continue

                self.signals_found += len(qualified)
                best = qualified[0]
                logger.info(
                    f"[SCANNER] ✓ {len(qualified)} confirmada(s) | "
                    f"Mejor: {best['asset']} | score={best['score']:.3f} | "
                    f"dir={best['direction']} | {best['reason'][:60]}"
                )
                self._notify("SCANNER_SCAN", f"{len(qualified)} señal(es) confirmada(s)",
                             data={"scan": self.last_scan_result, "best": best["asset"]})

                # Ejecutar el mejor candidato (fallback a 2do, 3ro...)
                for candidate in qualified:
                    if not self.running:
                        break
                    executed = await asyncio.to_thread(self._try_execute, candidate)
                    if executed:
                        self.trades_executed += 1
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SCANNER] Error en scan loop: {e}", exc_info=True)
                await asyncio.sleep(CANDLE_INTERVAL)

        self.running = False
        logger.info("[SCANNER] Bucle finalizado.")

    def _confirm_hot(self, hot_candidates: List[Dict]) -> List[Dict]:
        """
        Confirmación ultraligera post-cierre.

        Usa el DataFrame cacheado del pre-scan: solo fetchea 50 velas
        (no 300) y reconstruye rápido. 50 es suficiente para RSI(14),
        BB(14) y ATR(14) pero ~6x más rápido que el fetch completo.
        """
        CONFIRM_HISTORY = 50

        def _confirm_one(candidate: Dict) -> Dict:
            asset = candidate["asset"]
            base = {
                "asset": asset, "qualifies": False,
                "reason": "", "score": 0.0, "df": None,
                "direction": None,
            }
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(
                        iq_service.get_candles, asset, CANDLE_INTERVAL, CONFIRM_HISTORY
                    )
                    raw = future.result(timeout=GET_CANDLES_TIMEOUT)

                if not raw or len(raw) < 20:
                    base["reason"] = "sin datos frescos"
                    return base

                df = build_dataframe(raw)
                if df.empty or len(df) < 20:
                    base["reason"] = "df insuficiente"
                    return base

                qualifies, reason = pre_qualify(df, asset=asset)
                if not qualifies:
                    base["reason"] = reason
                    return base

                direction, score, _sigs, resolution = _evaluate_strategies(df, asset=asset)

                return {
                    "asset": asset, "qualifies": True,
                    "reason": reason, "score": score,
                    "df": df, "direction": direction,
                    "resolution": resolution,
                }
            except Exception as e:
                logger.warning(f"  {asset}: error en confirmación → {e}")
                base["reason"] = f"confirm error: {e}"
                return base

        results = []
        with ThreadPoolExecutor(max_workers=len(hot_candidates)) as pool:
            futures = {pool.submit(_confirm_one, c): c["asset"] for c in hot_candidates}
            for future in futures:
                try:
                    results.append(future.result(timeout=GET_CANDLES_TIMEOUT + 2))
                except Exception as e:
                    asset = futures[future]
                    logger.warning(f"  {asset}: timeout confirmación → {e}")
                    results.append({
                        "asset": asset, "qualifies": False,
                        "reason": f"timeout: {e}", "score": 0.0,
                        "df": None, "direction": None,
                    })
        return results

    # ─── Escaneo de activos ───────────────────────────────────────────────────

    def _get_available_assets(self) -> List[str]:
        """
        Obtiene del broker los activos de tipo binary/turbo que están ABIERTOS ahora.
        Si el usuario especificó activos al iniciar, filtra solo los que están abiertos.
        Si no especificó nada, devuelve hasta MAX_ASSETS_PER_SCAN activos en rotación aleatoria.
        """
        try:
            all_assets = iq_service.get_assets()
            # Filtra por tipo (binary o turbo) y estado abierto — no por nombre
            open_assets = [
                a["name"] for a in all_assets
                if a.get("open") and a.get("type") in ("binary", "turbo")
            ]

            if self.assets:
                # Intersectar la lista del usuario con los que están abiertos
                filtered = [a for a in self.assets if a in open_assets]
                logger.info(
                    f"[SCANNER] Activos del usuario: {len(self.assets)} | "
                    f"Abiertos ahora: {len(filtered)}"
                )
                random.shuffle(filtered)
                return filtered[:MAX_ASSETS_PER_SCAN]
            else:
                # Rotar aleatoriamente para cubrir todos los activos a lo largo del tiempo
                random.shuffle(open_assets)
                batch = open_assets[:MAX_ASSETS_PER_SCAN]
                logger.info(
                    f"[SCANNER] Activos abiertos: {len(open_assets)} | "
                    f"Escaneando este ciclo: {len(batch)} → {batch}"
                )
                return batch

        except Exception as e:
            logger.warning(f"[SCANNER] Error obteniendo activos: {e}")
            # Fallback: devolver la lista original limitada
            fallback = list(self.assets)
            random.shuffle(fallback)
            return fallback[:MAX_ASSETS_PER_SCAN]

    def _scan_all_assets(self) -> List[Dict]:
        """
        1. Consulta al broker qué activos OTC están abiertos ahora
        2. Solo escanea esos — nunca activos cerrados o inexistentes
        3. Escanea en paralelo (máx 5 hilos) para terminar rápido
        """
        candidates = self._get_available_assets()

        if not candidates:
            logger.warning("[SCANNER] No hay activos abiertos disponibles en este momento.")
            return []

        results = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(self._scan_one, asset): asset for asset in candidates}
            for future in futures:
                if not self.running:
                    break
                try:
                    result = future.result(timeout=GET_CANDLES_TIMEOUT + 3)
                    results.append(result)
                except Exception as e:
                    asset = futures[future]
                    logger.warning(f"  {asset}: timeout en scan paralelo → {e}")
                    results.append({
                        "asset": asset, "qualifies": False,
                        "reason": f"timeout: {e}", "score": 0.0,
                        "df": None, "direction": None, "conn_error": True,
                    })

        return results

    def _get_candles_safe(self, asset: str) -> List[Dict]:
        """
        Llama a get_candles con timeout estricto.
        Si la librería se bloquea (WS caído), aborta en GET_CANDLES_TIMEOUT segundos
        en lugar de quedarse bloqueada indefinidamente.
        """
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(iq_service.get_candles, asset, CANDLE_INTERVAL, HISTORY_COUNT)
            try:
                return future.result(timeout=GET_CANDLES_TIMEOUT)
            except FuturesTimeout:
                logger.warning(
                    f"  {asset}: ✗ timeout {GET_CANDLES_TIMEOUT}s — WS bloqueado, abortando"
                )
                return []

    def _scan_one(self, asset: str) -> Dict:
        """Evalúa un activo: fetch candles → pre_qualify → score → dirección."""
        base = {
            "asset": asset, "qualifies": False,
            "reason": "", "score": 0.0, "df": None, "direction": None,
            "conn_error": False,
        }
        try:
            raw = self._get_candles_safe(asset)
            if not raw:
                base["reason"] = "sin datos (WS caído o timeout)"
                base["conn_error"] = True
                logger.info(f"  {asset}: ✗ sin datos — WS caído o timeout")
                return base

            if len(raw) < 30:
                base["reason"] = f"datos insuficientes ({len(raw)} velas)"
                logger.info(f"  {asset}: ✗ {base['reason']}")
                return base

            df = build_dataframe(raw)
            if df.empty or len(df) < 30:
                base["reason"] = "df vacío"
                return base

            qualifies, reason = pre_qualify(df, asset=asset)
            if not qualifies:
                logger.info(f"  {asset}: ✗ {reason[:60]}")
                base["reason"] = reason
                return base

            direction, score, _sigs, resolution = _evaluate_strategies(df, asset=asset)

            logger.info(
                f"  {asset}: ✓ dir={direction} score={score:.3f} | {reason[:55]}"
            )
            return {
                "asset":     asset,
                "qualifies": True,
                "reason":    reason,
                "score":     score,
                "df":        df,
                "direction": direction,
                "resolution": resolution,
            }

        except Exception as e:
            logger.warning(f"  {asset}: error → {e}")
            base["reason"] = f"error: {e}"
            return base

    # ─── Ejecución del trade ──────────────────────────────────────────────────

    def _try_execute(self, candidate: Dict) -> bool:
        """
        Aplica filtros de régimen, valida con ML y ejecuta si supera el umbral.
        Devuelve True si se ejecutó el trade.

        Flujo:
          1. Direction check
          2. Contexto de mercado (payout, hora, weekday)
          3. check_all_filters (13 filtros de régimen)
          4. ML validation
          5. Guardia de tiempo
          6. Ejecución
        """
        asset     = candidate["asset"]
        df        = candidate["df"]
        direction = candidate["direction"]

        # ── Telemetría de estrategias (solo en flujo de decisión final) ───────
        self._check_telemetry_reset()
        resolution = candidate.get("resolution", "no_signal")
        if resolution in self._strategy_telemetry:
            self._strategy_telemetry[resolution] += 1

        if direction is None:
            logger.info(f"[SCANNER] {asset}: dirección indeterminada ({resolution}). Skip.")
            return False

        # ── Contexto de mercado (requerido por filtros Y por persistencia) ────
        try:
            last      = df.iloc[-1]
            ts_time   = last["time"]
            hour_utc  = int(ts_time.hour)
            weekday   = int(ts_time.weekday())

            winrate_map  = get_winrate_by_hour(asset=asset)
            winrate_hour = winrate_map.get(hour_utc, 0.5)
            payout       = iq_service.get_payout(asset, self.asset_type) or 0.80
        except Exception as e:
            logger.warning(f"[SCANNER] {asset}: error contexto → {e}")
            return False

        # ── Defaults para variables usadas más adelante en logging/persistencia ─
        proba  = 0.5
        pr_pct = 50

        # ── Determinar límites según modo ML ─────────────────────────────────
        if ML_DISABLED_MODE:
            asset_loss_limit = ML_DISABLED_MAX_ASSET_LOSSES
            daily_loss_limit = ML_DISABLED_MAX_DAILY_LOSSES
            trades_limit     = ML_DISABLED_MAX_DAILY_TRADES
        else:
            asset_loss_limit = MAX_ASSET_DAILY_LOSSES
            daily_loss_limit = MAX_DAILY_LOSSES
            trades_limit     = MAX_TRADES_PER_DAY

        # ── Filtros de régimen OTC (14 filtros via check_all_filters) ─────────
        regime = check_all_filters(
            df=df,
            asset=asset,
            trade_log=self.trade_log,
            payout=payout,
            direction=direction,
            max_asset_losses=asset_loss_limit,
            max_daily_losses=daily_loss_limit,
            max_trades=trades_limit,
        )
        if not regime.allow:
            logger.info(f"[SCANNER] {asset}: ✗ {regime.filter_name}: {regime.reason}")
            if regime.auto_shutdown:
                logger.warning(f"[SCANNER][AUTOSHUTDOWN] {regime.reason}")
                self._notify("BOT_AUTOSHUTDOWN", regime.reason, data={"filter": regime.filter_name})
                self.running = False
            return False

        # ── Validación ML o Score de estrategia ───────────────────────────────
        if ML_DISABLED_MODE:
            # Sin ML: validar por score de estrategia
            score = candidate.get("score", 0.0)
            logger.info(
                f"[SCANNER] {asset} | {direction} | ML_DISABLED | "
                f"score={score:.3f} | umbral={ML_DISABLED_MIN_SCORE}"
            )
            if score < ML_DISABLED_MIN_SCORE:
                logger.info(
                    f"[SCANNER] {asset}: score {score:.3f} < {ML_DISABLED_MIN_SCORE}. "
                    "Rechazado (ML_DISABLED_MODE)."
                )
                return False
        else:
            # Con ML: validación estándar
            proba  = 0.5
            pr_pct = 50

            try:
                from database import fetch_trades
                closed_count = len([
                    t for t in fetch_trades(limit=9999, path=DB_PATH)
                    if t.get("result") in ("WIN", "LOSS")
                ])
            except Exception:
                closed_count = 0

            ml_active = ml_classifier.is_loaded() and closed_count >= ML_MIN_TRADES

            if ml_active:
                try:
                    features = extract_features(
                        df, payout=payout, winrate_hour=winrate_hour,
                        direction=direction, expiry_min=self.expiry_min,
                    )
                    if features is None:
                        logger.info(f"[SCANNER] {asset}: features None. Skip.")
                        return False
                    probas    = ml_classifier.predict_proba(features)
                    proba_key = "call_proba" if direction == "CALL" else "put_proba"
                    proba     = probas.get(proba_key, 0.5)
                    pr_pct    = int(round(proba * 100))
                except Exception as e:
                    logger.warning(f"[SCANNER] {asset}: error ML → {e}")

                logger.info(
                    f"[SCANNER] {asset} | {direction} | ML proba={pr_pct}% | umbral={MIN_PROBABILITY}%"
                )
                if pr_pct < MIN_PROBABILITY:
                    logger.info(f"[SCANNER] {asset}: rechazado por ML ({pr_pct}% < {MIN_PROBABILITY}%)")
                    return False
            else:
                logger.info(
                    f"[SCANNER] {asset}: ML requerido pero no cargado. No se opera."
                )
                return False

        # ── Guardia de tiempo: no entrar si ya pasaron muchos segundos de la vela ─
        secs_into_candle = time.time() % CANDLE_INTERVAL
        if secs_into_candle > MAX_ENTRY_SECS:
            logger.warning(
                f"[SCANNER] {asset}: señal válida pero llegamos tarde "
                f"({secs_into_candle:.0f}s > {MAX_ENTRY_SECS}s en la vela). Skip."
            )
            return False

        # ── Ejecutar orden ────────────────────────────────────────────────────
        if self.mode == "paper":
            ok, order_id = True, -1
            logger.info(
                f"[SCANNER][PAPER] {asset} {direction} ${self.amount:.2f} {self.expiry_min}m"
            )
        else:
            ok, order_id = iq_service.buy_binary(
                asset, self.amount, direction.lower(), self.expiry_min
            )

        if not ok:
            logger.warning(f"[SCANNER] {asset}: orden rechazada por el broker.")
            return False

        entry_price = float(last["close"])
        ts_now      = datetime.now(timezone.utc).isoformat()

        # Cooldown global = expiración + 10s buffer
        self._cooldown_until = time.time() + self.expiry_min * 60 + 10

        logger.info(
            f"[SCANNER][TRADE ✓] {asset} {direction} | pr={pr_pct}% | "
            f"entry={entry_price:.5f} | ex={self.expiry_min}m | order={order_id}"
        )

        # Persistir en trades.db (retorna el id de la fila)
        db_id = self._persist(
            asset, direction, proba, entry_price, payout,
            order_id, ts_now, hour_utc, weekday, candidate["reason"], df
        )

        # Log en memoria
        record = {
            "db_id":     db_id,
            "ts":        ts_now,
            "asset":     asset,
            "direction": direction,
            "entry":     entry_price,
            "proba":     pr_pct,
            "reason":    candidate["reason"][:80],
            "result":    "PENDING",
            "order_id":  order_id,
        }
        self.trade_log = [record] + self.trade_log[:49]
        self._notify(
            "SCANNER_TRADE",
            f"Trade ejecutado: {asset} {direction} pr={pr_pct}%",
            data=record,
        )

        # Programar verificación del resultado tras la expiración
        if db_id and order_id != -1:
            self._schedule_coro(
                self._check_result_task(db_id, order_id, self.amount, self.expiry_min, asset, direction, record)
            )

        return True

    # ─── Persistencia ─────────────────────────────────────────────────────────

    def _persist(
        self,
        asset: str,
        direction: str,
        proba: float,
        entry_price: float,
        payout: float,
        order_id: int,
        ts: str,
        hour_utc: int,
        weekday: int,
        reason: str,
        df: pd.DataFrame,
    ) -> Optional[int]:
        """Guarda el trade en la BD y retorna el id de la fila."""
        last = df.iloc[-1]
        bb_upper = float(last["bb_upper"])
        bb_lower = float(last["bb_lower"])
        bb_range = bb_upper - bb_lower
        bb_pct_b = (entry_price - bb_lower) / bb_range if bb_range > 1e-10 else 0.5

        prng = compute_prng_features(df)

        record = {
            "timestamp":       ts,
            "asset":           asset,
            "direction":       direction,
            "expiry_min":      self.expiry_min,
            "mode":            self.mode,
            "price":           entry_price,
            "rsi":             float(last.get("rsi", 50)),
            "bb_pct_b":        round(bb_pct_b, 4),
            "bb_width_pct":    float(last.get("bb_width", 0)),
            "vol_rel":         float(last.get("vol_rel", 1.0)),
            "ema20":           float(last.get("ema20", 0)),
            "ema200":          float(last.get("ema200", 0)),
            "hour_utc":        hour_utc,
            "weekday":         weekday,
            "predicted_proba": proba,
            "ai_reasoning":    f"[scanner] {reason[:200]}",
            "order_id":        order_id,
            "open_price":      entry_price,
            "payout":          payout,
            "result":          "PENDING",
            # ── Features PRNG ─────────────────────────────────────────────────
            **prng,
        }
        try:
            return insert_trade(record, path=DB_PATH)
        except Exception as e:
            logger.warning(f"[SCANNER] Error persistiendo trade: {e}")
            return None

    # ─── Verificador de resultados ────────────────────────────────────────────

    async def _check_result_task(
        self,
        db_id: int,
        order_id: int,
        amount: float,
        expiry_min: int,
        asset: str,
        direction: str,
        mem_record: Dict,
    ) -> None:
        """
        Espera la expiración de la orden y consulta el resultado al broker.
        Actualiza la BD y el log en memoria con WIN/LOSS.
        """
        wait_secs = expiry_min * 60 + 5   # buffer de 5s tras la expiración
        logger.info(f"[RESULT] Esperando {wait_secs}s para verificar order {order_id} ({asset})")
        await asyncio.sleep(wait_secs)

        try:
            profit = await asyncio.to_thread(iq_service.check_win, order_id)
        except Exception as e:
            logger.error(f"[RESULT] Error consultando order {order_id}: {e}")
            return

        if profit is None:
            logger.warning(f"[RESULT] No se pudo obtener resultado para order {order_id}")
            return

        result    = "WIN" if profit > 0 else ("LOSS" if profit < 0 else "TIE")
        closed_at = datetime.now(timezone.utc).isoformat()

        # ── Actualizar BD ─────────────────────────────────────────────────────
        try:
            update_trade_result_simple(db_id, result, profit, closed_at, path=DB_PATH)
        except Exception as e:
            logger.error(f"[RESULT] Error actualizando BD trade {db_id}: {e}")

        # ── Actualizar log en memoria ─────────────────────────────────────────
        mem_record["result"] = result
        mem_record["profit"] = profit

        sign = "+" if profit >= 0 else ""
        logger.info(
            f"[RESULT ✓] {asset} {direction} → {result} | "
            f"P&L: {sign}{profit:.2f} | order={order_id}"
        )

        # ── Notificar al frontend ─────────────────────────────────────────────
        self._notify(
            "TRADE_RESULT",
            f"{asset} {direction} → {result}  {sign}{profit:.2f}",
            data={
                "db_id":     db_id,
                "order_id":  order_id,
                "asset":     asset,
                "direction": direction,
                "result":    result,
                "profit":    profit,
            },
        )

        # ── Auto-aprendizaje: analizar pérdida ───────────────────────────────
        if result == "LOSS":
            self._schedule_coro(
                self._analyze_loss(db_id, asset, direction, mem_record.get("entry", 0.0), expiry_min)
            )

    async def _analyze_loss(
        self, db_id: int, asset: str, direction: str, entry_price: float, expiry_min: int
    ) -> None:
        """
        Analiza una pérdida para auto-aprendizaje.

        1. Espera 5 velas adicionales después del cierre
        2. Fetcha velas durante y después del trade
        3. Calcula max_adverse_pips, max_favorable_pips, price_after_5
        4. Clasifica el tipo de pérdida
        5. Guarda en la BD
        """
        # Esperar 5 velas extra para ver qué pasó después
        await asyncio.sleep(5 * CANDLE_INTERVAL + 5)

        try:
            # Fetch velas que cubran el trade + 5 velas post
            total_candles = expiry_min + 10
            raw = await asyncio.to_thread(
                iq_service.get_candles, asset, CANDLE_INTERVAL, total_candles
            )
            if not raw or len(raw) < expiry_min + 5:
                logger.warning(f"[LEARN] {asset}: datos insuficientes para análisis post-trade")
                return

            closes = [float(c.get("close", 0)) for c in raw]
            highs = [float(c.get("high", 0)) for c in raw]
            lows = [float(c.get("low", 0)) for c in raw]

            # Las velas del trade son las últimas (expiry_min + 5 + algo de buffer)
            # Buscamos la vela de entrada por proximidad de precio
            trade_start = -(expiry_min + 5 + 1)
            trade_end = -6  # 5 velas antes del final
            post_start = -5  # últimas 5 velas = post-trade

            trade_highs = highs[trade_start:trade_end] if abs(trade_start) <= len(highs) else highs
            trade_lows = lows[trade_start:trade_end] if abs(trade_start) <= len(lows) else lows

            if not trade_highs or entry_price == 0:
                return

            # Max adverse pips (peor momento en contra)
            if direction == "CALL":
                max_adverse = entry_price - min(trade_lows)
                max_favorable = max(trade_highs) - entry_price
            else:  # PUT
                max_adverse = max(trade_highs) - entry_price
                max_favorable = entry_price - min(trade_lows)

            # Normalizar a pips
            if entry_price > 10:  # JPY pairs
                max_adverse_pips = max_adverse * 100
                max_favorable_pips = max_favorable * 100
            else:
                max_adverse_pips = max_adverse * 10000
                max_favorable_pips = max_favorable * 10000

            # Precio 5 velas después del cierre
            price_after_5 = closes[-1] if closes else None

            # Clasificar tipo de pérdida
            loss_type = self._classify_loss(
                direction, entry_price, price_after_5,
                max_adverse_pips, max_favorable_pips
            )

            # Guardar en BD
            update_trade_post_analysis(
                trade_id=db_id,
                loss_type=loss_type,
                price_after_5=price_after_5,
                max_adverse_pips=round(max_adverse_pips, 2),
                max_favorable_pips=round(max_favorable_pips, 2),
            )

            logger.info(
                f"[LEARN] {asset} loss analizada: {loss_type} | "
                f"adverse={max_adverse_pips:.1f}p favorable={max_favorable_pips:.1f}p | "
                f"price_after_5={price_after_5}"
            )

        except Exception as e:
            logger.error(f"[LEARN] Error analizando pérdida trade {db_id}: {e}")

    @staticmethod
    def _classify_loss(
        direction: str, entry_price: float, price_after_5: Optional[float],
        max_adverse_pips: float, max_favorable_pips: float
    ) -> str:
        """
        Clasifica el tipo de pérdida para auto-aprendizaje.

        Tipos:
          - spread:            perdió por margen mínimo (< 2 pips)
          - entrada_prematura: el precio eventualmente fue a favor (after_5 confirma)
          - falsa_reversion:   nunca tuvo movimiento favorable significativo
          - tendencia_fuerte:  gran movimiento en contra (> 10 pips)
        """
        # Spread: perdió por muy poco
        if max_adverse_pips < 2.0 and max_favorable_pips < 2.0:
            return "spread"

        # Tendencia fuerte: gran movimiento en contra
        if max_adverse_pips > 10.0:
            return "tendencia_fuerte"

        # Entrada prematura: después de 5 velas el precio fue a nuestro favor
        if price_after_5 is not None:
            if direction == "CALL" and price_after_5 > entry_price:
                return "entrada_prematura"
            if direction == "PUT" and price_after_5 < entry_price:
                return "entrada_prematura"

        # Falsa reversión: nunca tuvo movimiento favorable significativo
        if max_favorable_pips < 1.5:
            return "falsa_reversion"

        return "falsa_reversion"

    # ─── Notificaciones ───────────────────────────────────────────────────────

    def _notify(self, event_type: str, message: str, data: Dict = None) -> None:
        if self.on_notification:
            payload = {"type": event_type, "message": message, "data": data or {}}
            try:
                self._schedule_coro(self.on_notification(payload))
            except Exception:
                pass

    def _schedule_coro(self, coro) -> None:
        """Encola una corutina en el event loop principal de forma segura desde threads."""
        if not self.loop:
            return
        try:
            loop = asyncio.get_running_loop()
            if loop is self.loop:
                self.loop.create_task(coro)
            else:
                asyncio.run_coroutine_threadsafe(coro, self.loop)
        except RuntimeError:
            asyncio.run_coroutine_threadsafe(coro, self.loop)


# ─── Evaluación consolidada de estrategias ───────────────────────────────────

def _evaluate_strategies(
    df: pd.DataFrame,
    asset: str = "",
) -> Tuple[Optional[str], float, List[StrategySignal], str]:
    """
    Evalúa las 4 estrategias en paralelo y aplica arbitraje de conflictos.

    Reemplaza _infer_direction (cascada) + _score_signal (duplicación).
    Cada estrategia se evalúa independientemente. Si hay conflicto de
    dirección, se cancela la operación.

    Returns:
        (direction, score, signals, resolution)
        - direction: "CALL"/"PUT" o None si conflicto/nada activo
        - score: score del ganador (0.0 si None)
        - signals: lista completa de StrategySignal para auditoría
        - resolution: "single_match" | "multiple_match" | "conflict_cancel" | "no_signal"
    """
    last    = df.iloc[-1]
    rsi     = float(last["rsi"])
    price   = float(last["close"])
    open_   = float(last["open"])
    bb_mid  = float(last["bb_mid"])
    bb_up   = float(last["bb_upper"])
    bb_low  = float(last["bb_lower"])
    bb_rng  = bb_up - bb_low
    vol_rel = float(last.get("vol_rel", 1.0))

    signals: List[StrategySignal] = []

    # ── 1. BB 2-Candle Reversal ──────────────────────────────────────────────
    bb2_ok, bb2_dir, _ = detect_bb_two_candle_reversal(df)
    if bb2_ok and bb2_dir:
        rsi_extremity = (25 - rsi) / 25 if rsi < 25 else (rsi - 75) / 25 if rsi > 75 else 0.0
        s = round(min(0.70 + min(rsi_extremity, 1.0) * 0.25, 1.0), 4)
        signals.append(StrategySignal(True, bb2_dir, s, "bb_2candle"))
    else:
        signals.append(StrategySignal(False, None, 0.0, "bb_2candle"))

    # ── 2. BB Body Reversal — DEPRECADA (Tarea 2.3) ────────────────────────
    # No participa en decisión. Se registra como phantom en el log JSONL.
    # Rendimiento histórico: 25% WR (1W/3L en 4 trades).
    # Solo PUT, asimetría no validable. Deprecación instrumentada.
    signals.append(StrategySignal(False, None, 0.0, "bb_body"))

    # ── 3. Streak Reversal ───────────────────────────────────────────────────
    streak = get_streak_info(df)
    if streak["is_extreme"]:
        pct_f = streak["percentile"] / 100
        len_f = min(streak["current_length"], 8) / 8
        s = round(min(pct_f * 0.75 + len_f * 0.25, 1.0), 4)
        streak_dir = "CALL" if streak["direction"] == "bear" else "PUT"
        signals.append(StrategySignal(True, streak_dir, s, "streak"))
    else:
        signals.append(StrategySignal(False, None, 0.0, "streak"))

    # ── 4. RSI Clásico + BB (umbrales endurecidos: 25/75, zone 10%) ─────────
    pct_b = (price - bb_low) / bb_rng if bb_rng > 1e-10 else 0.5
    if rsi < 25:
        rsi_f = (25 - rsi) / 25
        bb_f  = max(0.0, 0.10 - pct_b) / 0.10
        s = round(min(0.40 + rsi_f * 0.30 + bb_f * 0.30, 1.0), 4)
        signals.append(StrategySignal(True, "CALL", s, "rsi_classical"))
    elif rsi > 75:
        rsi_f = (rsi - 75) / 25
        bb_f  = max(0.0, pct_b - 0.90) / 0.10
        s = round(min(0.40 + rsi_f * 0.30 + bb_f * 0.30, 1.0), 4)
        signals.append(StrategySignal(True, "PUT", s, "rsi_classical"))
    else:
        signals.append(StrategySignal(False, None, 0.0, "rsi_classical"))

    # ── Arbitraje ────────────────────────────────────────────────────────────
    active = [s for s in signals if s.active]

    if not active:
        resolution = "no_signal"
        direction, winner_score = None, 0.0
    else:
        directions = {s.direction for s in active}
        if len(directions) == 1:
            winner = max(active, key=lambda s: s.score)
            resolution = "single_match" if len(active) == 1 else "multiple_match"
            direction, winner_score = winner.direction, winner.score
        else:
            resolution = "conflict_cancel"
            direction, winner_score = None, 0.0

    # ── Instrumentación phantom BB Body (deprecada pero observada) ──────────
    phantom_signal = None
    try:
        put_ok, _ = detect_bb_body_reversal(df)
        call_ok, _ = detect_bb_body_reversal_call(df)
        if put_ok and call_ok:
            phantom_signal = "BOTH_phantom"
        elif put_ok:
            phantom_signal = "PUT_phantom"
        elif call_ok:
            phantom_signal = "CALL_phantom"
    except Exception:
        pass

    # ── Log JSONL ────────────────────────────────────────────────────────────
    try:
        winner_data = None
        if direction is not None:
            w = max(active, key=lambda s: s.score)
            winner_data = {"strategy": w.strategy_name, "direction": w.direction, "score": w.score}

        entry = {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "asset": asset,
            "resolution": resolution,
            "active_count": len(active),
            "winner": winner_data,
            "all_signals": [
                {"strategy": s.strategy_name, "active": s.active,
                 "direction": s.direction, "score": s.score}
                for s in signals
            ],
            "conflict_detail": (
                ", ".join(f"{s.strategy_name}={s.direction}" for s in active)
                if resolution == "conflict_cancel" else None
            ),
            "phantom_signals": {
                "bb_body": phantom_signal,
            },
        }
        _decision_logger.info(json.dumps(entry, separators=(",", ":")))
    except Exception:
        pass

    # ── Log principal ────────────────────────────────────────────────────────
    if resolution == "conflict_cancel":
        detail = ", ".join(f"{s.strategy_name}={s.direction}" for s in active)
        logger.info(f"[DIRECTION] conflict_cancel: {detail}")
    elif active:
        w = max(active, key=lambda s: s.score)
        logger.info(
            f"[DIRECTION] {resolution}: {len(active)} estrategia(s) → {direction} "
            f"(score={winner_score:.3f}, ganadora={w.strategy_name})"
        )

    return direction, winner_score, signals, resolution


# ─── Singleton ────────────────────────────────────────────────────────────────

asset_scanner = AssetScanner()
