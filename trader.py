"""
trader.py – Orquestador del Bot de Trading Autónomo v3

DEPRECATED: This module is deprecated as of remediation/v1.

The official trading pipeline is asset_scanner.py.
This module uses a different ML threshold (78% vs scanner's 55%) and
a different decision pipeline that has not been validated under the
current remediation plan.

To use this module, the deprecation guard must be explicitly disabled
by setting ALLOW_DEPRECATED_TRADERS = True in iqservice.py.
This should only be done with full understanding of the risks.

See CHANGELOG_REMEDIATION.md for details.
"""

import warnings
from iqservice import ALLOW_DEPRECATED_TRADERS

if not ALLOW_DEPRECATED_TRADERS:
    raise ImportError(
        f"{__name__} is deprecated and disabled during remediation. "
        "Use asset_scanner.py instead. "
        "If you have a valid reason to use this module, set "
        "ALLOW_DEPRECATED_TRADERS=True in iqservice.py with documented justification."
    )

warnings.warn(
    f"{__name__} is deprecated. Use asset_scanner.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import json
import os
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from indicators import build_dataframe, get_streak_info, pre_qualify
from iqservice import iq_service
from ml_classifier import ml_classifier, extract_features
from regime_filter import check_all_filters

# ai_brain es opcional: se importa solo como fallback cuando no hay modelo ML
try:
    from ai_brain import get_ai_decision as _llm_decision
    _LLM_AVAILABLE = True
except ImportError:
    _llm_decision = None
    _LLM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─── Constantes de configuración ─────────────────────────────────────────────

CANDLE_INTERVAL  = 60    # segundos por vela (1 minuto)
HISTORY_COUNT    = 300   # velas históricas a solicitar por ciclo
MIN_PROBABILITY  = 78    # umbral mínimo de confianza para ejecutar (%)
LOOP_POLL_SECS   = 5     # intervalo de polling para detectar nueva vela
STREAM_INIT_SECS = 2     # tiempo de inicialización del stream de IQ Option


# ─── Clase Principal del Bot ──────────────────────────────────────────────────

class TradingBot:
    """
    Bot de trading autónomo con:
    - Filtro de pre-calificación (RSI extremo + toque de BB) para despertar a la IA
    - Doble gatillo de ejecución (pr ≥ 85% y alineación con EMA200)
    - Cooldown automático tras cada operación
    - Log detallado en consola de cada decisión y resultado
    """

    def __init__(self):
        self.running: bool          = False
        self.asset: str             = ""
        self.asset_type: str        = "binary"
        self.amount: float          = 1000.0
        self._task: Optional[asyncio.Task] = None
        self._last_candle_ts: Optional[int] = None
        self._cooldown_until: float = 0.0
        self.trade_log: List[Dict]  = []
        self.last_decision: Optional[Dict] = None
        self.last_analysis_time: Optional[str] = None
        # Estadísticas del ciclo
        self.total_cycles: int      = 0
        self.prequalified: int      = 0
        self.skipped_preq: int      = 0
        self.skipped_gates: int     = 0
        self.trades_executed: int   = 0
        self.signal_history: List[Dict] = []
        self.on_notification = None
        self._last_payout: float = 0.80

    # ─── Interfaz Pública de Control ─────────────────────────────────────────

    def start(self, asset: str, asset_type: str, amount: float) -> bool:
        """
        Inicia el bot en una tarea asyncio de background.

        Args:
            asset:      Activo a operar (e.g., "EURUSD")
            asset_type: "binary", "turbo" o "digital"
            amount:     Monto en dólares por operación
        """
        if self.running:
            logger.warning("Bot ya en ejecución.")
            return False
        if not iq_service.is_connected():
            logger.error("Sin conexión a IQ Option. Conectar primero.")
            return False

        self.asset         = asset
        self.asset_type    = asset_type
        self.amount        = amount
        self.running       = True
        self._last_candle_ts = None
        self._cooldown_until = 0.0
        self.trade_log     = []
        self.total_cycles  = 0
        self.prequalified  = 0
        self.skipped_preq  = 0
        self.skipped_gates = 0
        self.trades_executed = 0
        self.last_decision = None
        self.last_analysis_time = None
        self.signal_history = []

        # Intentar cargar modelo ML al inicio
        if not ml_classifier.is_loaded():
            ml_classifier.load()

        engine = "ML(LightGBM)" if ml_classifier.is_loaded() else (
            "LLM(Ollama)" if _LLM_AVAILABLE else "pre_qualify_only"
        )

        self._task = asyncio.create_task(self._run_loop())
        _log_banner(
            f"BOT INICIADO | {asset} | {asset_type.upper()} | ${amount:.2f}/op | "
            f"Motor: {engine} | Umbral: {MIN_PROBABILITY}%"
        )
        return True

    def stop(self) -> None:
        """Detiene el bot después del ciclo actual."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        iq_service.stop_stream(self.asset, CANDLE_INTERVAL)
        _log_banner(
            f"BOT DETENIDO | Ciclos: {self.total_cycles} | "
            f"Pre-cal: {self.prequalified} | Operaciones: {self.trades_executed}"
        )

    def get_status(self) -> Dict:
        """Estado completo del bot para el endpoint /bot/status."""
        wins   = sum(1 for t in self.trade_log if t.get("result") == "WIN")
        total  = len(self.trade_log)
        return {
            "running":            self.running,
            "asset":              self.asset,
            "asset_type":         self.asset_type,
            "amount":             self.amount,
            "model":              "ML(LightGBM)" if ml_classifier.is_loaded() else "LLM(Ollama fallback)",
            "min_probability":    MIN_PROBABILITY,
            "last_decision":      self.last_decision,
            "last_analysis_time": self.last_analysis_time,
            "cooldown_remaining": round(max(0.0, self._cooldown_until - time.time()), 1),
            "stats": {
                "total_cycles":     self.total_cycles,
                "prequalified":     self.prequalified,
                "skipped_preq":     self.skipped_preq,
                "skipped_gates":    self.skipped_gates,
                "trades_executed":  self.trades_executed,
                "wins":   wins,
                "losses": total - wins,
                "win_rate": f"{wins/total*100:.1f}%" if total else "N/A",
            },
            "trade_log": self.trade_log[-20:],
            "signal_history": self.signal_history
        }

    # ─── Bucle Principal ──────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """
        Bucle de detección de nueva vela.

        Diseño de memoria: el DataFrame de la iteración anterior se descarta
        explícitamente antes de construir el nuevo (`del df`, `del raw_candles`).
        Esto es crítico en M5 con 16GB de memoria unificada: el LLM y el bot
        comparten el mismo pool de memoria y debemos minimizar la fragmentación.
        """
        logger.info("Bucle iniciando – esperando primera vela cerrada...")

        iq_service.start_stream(self.asset, CANDLE_INTERVAL)
        await asyncio.sleep(STREAM_INIT_SECS)

        while self.running:
            try:
                now               = time.time()
                current_candle_ts = int(now // CANDLE_INTERVAL * CANDLE_INTERVAL)

                # ── Primera iteración: establecer referencia temporal ──────────
                if self._last_candle_ts is None:
                    self._last_candle_ts = current_candle_ts
                    dt = datetime.fromtimestamp(current_candle_ts, tz=timezone.utc)
                    logger.info(f"Referencia temporal: {dt.strftime('%H:%M:%S')} UTC")
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                # ── Misma vela en formación → esperar ─────────────────────────
                if current_candle_ts <= self._last_candle_ts:
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                # ── Nueva vela cerrada ────────────────────────────────────────
                self._last_candle_ts = current_candle_ts
                self.total_cycles   += 1
                dt = datetime.fromtimestamp(current_candle_ts, tz=timezone.utc)
                logger.info(f"{'─'*60}")
                logger.info(
                    f"[VELA #{self.total_cycles}] {dt.strftime('%H:%M:%S')} UTC | "
                    f"{self.asset}"
                )

                # ── Cooldown activo ───────────────────────────────────────────
                if time.time() < self._cooldown_until:
                    remaining = self._cooldown_until - time.time()
                    logger.info(f"[COOLDOWN] {remaining:.0f}s restantes. Skip.")
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                # ── Obtener datos frescos (liberar DataFrame anterior) ─────────
                raw_candles = await asyncio.to_thread(
                    iq_service.get_candles, self.asset, CANDLE_INTERVAL, HISTORY_COUNT
                )

                if not raw_candles or len(raw_candles) < 205:
                    count = len(raw_candles) if raw_candles else 0
                    logger.warning(f"[DATOS] Insuficientes: {count}/205 velas. Skip.")
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                df = build_dataframe(raw_candles)
                del raw_candles  # ← liberar lista en memoria inmediatamente

                if df.empty:
                    logger.warning("[DATOS] DataFrame vacío. Skip.")
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                # ── Log de indicadores actuales ───────────────────────────────
                _log_indicators(df)

                # ── Filtros de régimen OTC ────────────────────────────────────
                payout = await asyncio.to_thread(iq_service.get_payout, self.asset, self.asset_type)
                self._last_payout = payout or 0.80  # guardado para _get_decision
                regime = check_all_filters(
                    df=df,
                    asset=self.asset,
                    trade_log=self.trade_log,
                    payout=payout,
                )
                if not regime.allow:
                    logger.info(f"[RÉGIMEN] ✗ {regime.filter_name}: {regime.reason}")
                    self._notify(
                        "REGIME_BLOCK",
                        f"Bloqueado: {regime.filter_name}",
                        data={"filter": regime.filter_name, "reason": regime.reason},
                    )
                    del df
                    if regime.auto_shutdown:
                        logger.warning(f"[AUTOSHUTDOWN] {regime.reason}")
                        self._notify("BOT_AUTOSHUTDOWN", regime.reason, data={"filter": regime.filter_name})
                        self.running = False
                        break
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                # ── Filtro de Pre-Calificación ────────────────────────────────
                qualifies, preq_reason = pre_qualify(df, asset=self.asset)

                if not qualifies:
                    logger.info(f"[PRE-CAL] ✗ {preq_reason} → IA no activada")
                    self.skipped_preq += 1
                    del df
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                self.prequalified += 1
                logger.info(f"[PRE-CAL] ✓ {preq_reason} → Activando motor de decisión...")
                self._notify("ANALYSIS", "Analizando mercado...", data={"reason": preq_reason})

                # ── Motor de decisión: ML primario / LLM fallback ─────────────
                t_start  = time.time()
                decision = await asyncio.to_thread(
                    self._get_decision, df
                )
                elapsed  = time.time() - t_start

                self.last_decision      = decision
                self.last_analysis_time = datetime.now(timezone.utc).isoformat()

                if decision is None:
                    logger.warning(f"[MOTOR] Sin decisión válida ({elapsed:.1f}s). Skip.")
                    del df
                    await asyncio.sleep(LOOP_POLL_SECS)
                    continue

                self._log_decision(decision, elapsed)

                # ── Doble Gatillo → Ejecución ─────────────────────────────────
                await self._evaluate_and_trade(df, decision)
                del df  # ← limpieza explícita al final del ciclo

            except asyncio.CancelledError:
                logger.info("[BOT] Cancelado por señal externa.")
                break
            except Exception as e:
                logger.error(f"[ERROR] Error inesperado en bucle: {e}", exc_info=True)
                await asyncio.sleep(LOOP_POLL_SECS)

        self.running = False
        logger.info("[BOT] Bucle finalizado.")

    # ─── Motor de Decisión (ML primario / LLM fallback) ──────────────────────

    def _get_decision(self, df) -> Optional[Dict]:
        """
        Obtiene la decisión de trading usando ML cuando el modelo está disponible,
        con fallback al LLM (Ollama) si no hay modelo entrenado.

        Devuelve un dict normalizado con claves: op, pr, ex, an, engine.
            op: "CALL" | "PUT" | "WAIT"
            pr: 0-100 (probabilidad como porcentaje entero)
            ex: minutos de expiración
            an: razón / descripción
            engine: "ml" | "llm"
        """
        from database import get_winrate_by_hour
        from indicators import get_streak_info

        # ── Motor ML ──────────────────────────────────────────────────────────
        if ml_classifier.is_loaded():
            try:
                last = df.iloc[-1]
                hour_utc = int(last.get("hour_utc", 0) or 0)
                winrate_map = get_winrate_by_hour(asset=self.asset)
                winrate_hour = winrate_map.get(hour_utc, 0.5)

                payout = self._last_payout if hasattr(self, "_last_payout") else 0.80

                probas = ml_classifier.predict_proba_from_df(
                    df, payout=payout, winrate_hour=winrate_hour
                )
                call_p = probas["call_proba"]
                put_p  = probas["put_proba"]

                # Elegir la dirección con mayor proba; usar expiración por defecto = 1m
                if call_p >= put_p:
                    op = "CALL"
                    pr = int(round(call_p * 100))
                else:
                    op = "PUT"
                    pr = int(round(put_p * 100))

                # Si ninguna supera el umbral, devolver WAIT para que GATE-1 lo filtre
                if pr < MIN_PROBABILITY:
                    op = "WAIT"

                logger.info(
                    f"[ML] CALL={call_p:.2%} PUT={put_p:.2%} → {op} pr={pr}%"
                )
                return {"op": op, "pr": pr, "ex": 1, "an": f"ML call={call_p:.2%} put={put_p:.2%}", "engine": "ml"}

            except Exception as exc:
                logger.warning(f"[ML] Error en predicción: {exc}. Usando fallback LLM.")

        # ── Fallback LLM ──────────────────────────────────────────────────────
        if _LLM_AVAILABLE and _llm_decision is not None:
            logger.info("[MOTOR] Modelo ML no disponible. Usando LLM (Ollama).")
            decision = _llm_decision(df, self.asset)
            if decision is not None:
                decision["engine"] = "llm"
            return decision

        logger.warning("[MOTOR] Ni ML ni LLM disponibles. Usando pre_qualify directo.")
        return None

    # ─── Doble Gatillo de Ejecución ───────────────────────────────────────────

    async def _evaluate_and_trade(self, df, decision: Dict) -> None:
        """
        Aplica los dos gatillos de seguridad antes de ejecutar cualquier orden.

        Gatillo 0: IA dice WAIT → abortar.
        Gatillo 1: Probabilidad de la IA ≥ MIN_PROBABILITY (85%).
        Gatillo 2: Dirección (CALL/PUT) alineada con la tendencia de EMA200.
        """
        op = decision["op"]
        pr = int(decision["pr"])
        ex = int(decision["ex"])

        signal_data = {
            "op": op,
            "pr": pr,
            "an": decision.get("an", ""),
            "status": "Pendiente"
        }

        # Gatillo 0: WAIT
        if op == "WAIT":
            logger.info(f"[GATE-0] IA dice WAIT (pr={pr}%). No se opera.")
            self.skipped_gates += 1
            signal_data["status"] = "Wait (IA indecisa)"
            self._add_signal(signal_data)
            self._notify("SIGNAL", "Señal: WAIT", data=signal_data)
            return

        # Gatillo 1: Probabilidad mínima
        if pr < MIN_PROBABILITY:
            logger.info(
                f"[GATE-1] ✗ pr={pr}% < mínimo {MIN_PROBABILITY}%. No se opera."
            )
            self.skipped_gates += 1
            signal_data["status"] = f"Bloqueado (Probabilidad {pr}% < {MIN_PROBABILITY}%)"
            self._add_signal(signal_data)
            self._notify("SIGNAL", f"Señal rechazada (pr={pr}%)", data=signal_data)
            return

        # Gatillo 2: Coherencia con señal OTC (racha extrema → reversión esperada)
        # En OTC, EMA200 no es un indicador de tendencia de mercado real.
        # Se verifica que la dirección del LLM sea coherente con la racha actual:
        # racha bajista extrema → esperar CALL (reversión); alcista extrema → PUT.
        # Sin racha extrema: se confía en el LLM sin filtro de tendencia.
        streak = get_streak_info(df)

        if streak["is_extreme"]:
            expected_call = streak["direction"] == "bear"   # reversión de bajista → CALL
            expected_put  = streak["direction"] == "bull"   # reversión de alcista → PUT

            if op == "CALL" and not expected_call:
                logger.info(
                    f"[GATE-2] ✗ IA dice CALL pero racha es {streak['direction'].upper()} "
                    f"(pct={streak['percentile']:.0f}%): no es reversión coherente. No se opera."
                )
                self.skipped_gates += 1
                signal_data["status"] = "Bloqueado (dirección contraria a racha OTC)"
                self._add_signal(signal_data)
                self._notify("SIGNAL", "Señal rechazada (racha OTC)", data=signal_data)
                return

            if op == "PUT" and not expected_put:
                logger.info(
                    f"[GATE-2] ✗ IA dice PUT pero racha es {streak['direction'].upper()} "
                    f"(pct={streak['percentile']:.0f}%): no es reversión coherente. No se opera."
                )
                self.skipped_gates += 1
                signal_data["status"] = "Bloqueado (dirección contraria a racha OTC)"
                self._add_signal(signal_data)
                self._notify("SIGNAL", "Señal rechazada (racha OTC)", data=signal_data)
                return
        else:
            logger.info(
                f"[GATE-2] Sin racha extrema (pct={streak['percentile']:.0f}%). "
                f"Confiando en motor ({decision.get('engine', '?')})."
            )

        # ── Ambos gatillos superados ──────────────────────────────────────────
        streak_info = (
            f"Racha {streak['direction'].upper()} {streak['current_length']}v "
            f"(pct={streak['percentile']:.0f}%)"
            if streak["is_extreme"] else f"motor={decision.get('engine', '?')}"
        )
        logger.info(
            f"[GATE-OK] ✓✓ Ambos gatillos superados | "
            f"{op} | pr={pr}% | ex={ex}m | {streak_info}"
        )
        logger.info(
            f"[ORDER] Ejecutando: {self.asset} {op} ${self.amount:.2f} {ex}m"
        )

        # Inyectar features ML en decision para que PaperTrader las persista en DB
        try:
            ml_feats = extract_features(
                df,
                payout=self._last_payout,
                winrate_hour=0.5,
                direction=op,
                expiry_min=ex,
            )
            decision["ml_features"] = ml_feats or {}
        except Exception:
            decision["ml_features"] = {}

        success, order_id = await asyncio.to_thread(self._place_order, op, ex)
        self.trades_executed += 1

        trade_record = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "asset":       self.asset,
            "op":          op,
            "amount":      self.amount,
            "expiry_min":  ex,
            "confidence":  pr,
            "engine":      decision.get("engine", "unknown"),
            "reason":      decision.get("an", ""),
            "executed":    success,
            "result":      "PENDING",
        }
        self.trade_log.append(trade_record)

        if success:
            self._cooldown_until = time.time() + (ex * CANDLE_INTERVAL) + 10
            logger.info(
                f"[ORDER] ✓ Orden aceptada. Cooldown: {ex}m + 10s buffer."
            )
            signal_data["status"] = "Ejecutada ✓"
            self._add_signal(signal_data)
            self._notify("EXECUTION", f"{op} en {self.asset} aceptada!", data=signal_data)
            
            # Lanzamos el monitor asíncrono para verificar victoria o derrota y calcular Pips
            asyncio.create_task(self._monitor_trade(order_id, op, ex, decision, df.iloc[-1].to_dict()))
        else:
            logger.error(f"[ORDER] ✗ IQ Option rechazó la orden.")
            signal_data["status"] = "Rechazada por IQ Option ✗"
            self._add_signal(signal_data)
            self._notify("ERROR", "IQ Option rechazó la orden", data=signal_data)

    # ─── Colocación de Órdenes ────────────────────────────────────────────────

    def _place_order(self, direction: str, expiry_minutes: int) -> Tuple[bool, int]:
        """
        Enruta la orden al método correcto de iqservice según el tipo de activo.
        """
        action = direction.lower()
        try:
            if self.asset_type == "digital":
                return iq_service.buy_digital(self.asset, self.amount, action, expiry_minutes)
            else:
                return iq_service.buy_binary(self.asset, self.amount, action, expiry_minutes)
        except Exception as e:
            logger.error(f"[ORDER] Excepción al colocar orden: {e}", exc_info=True)
            return False, -1

    async def _monitor_trade(self, order_id: int, op: str, ex: int, decision: Dict, last_candle: Dict):
        """Monitorea el resultado y los pips perdidos en background preventivamente."""
        logger.info(f"[MONITOR] Iniciando monitoreo de orden {order_id}...")
        profit = 0.0
        details = None

        # Timeout: tiempo de expiración + 2 minutos de margen
        max_wait_secs = ex * 60 + 120
        started_at = asyncio.get_event_loop().time()
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 15

        # Polling del resultado de IQ Option
        while True:
            elapsed = asyncio.get_event_loop().time() - started_at
            if elapsed > max_wait_secs:
                logger.warning(f"[MONITOR] Timeout ({max_wait_secs}s) esperando resultado de orden {order_id}. Abortando monitoreo.")
                return
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(f"[MONITOR] {MAX_CONSECUTIVE_ERRORS} errores consecutivos monitoreando orden {order_id}. Abortando.")
                return
            try:
                data = await asyncio.to_thread(iq_service.api.get_async_order, order_id)
                consecutive_errors = 0  # reset en polling exitoso
                if self.asset_type == "digital":
                    if data and "position-changed" in data and data["position-changed"]:
                        msg = data["position-changed"]["msg"]
                        if msg and msg.get("status") == "closed":
                            profit = float(msg.get("close_profit", 0) - msg.get("invest", 0)) if msg.get("close_reason") == "expired" else float(msg.get("pnl_realized", 0))
                            details = msg
                            break
                else: # Binary
                    if data and "option-closed" in data and data["option-closed"]:
                        msg = data["option-closed"]["msg"]
                        profit = float(msg.get("profit_amount", 0) - msg.get("amount", 0))
                        details = msg
                        break
            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"[MONITOR] Error #{consecutive_errors} consultando orden {order_id}: {e}")
            await asyncio.sleep(1.0)

        logger.info(f"[MONITOR] Orden {order_id} cerrada. Profit: {profit:.2f}")

        if profit < 0 and details is not None:
            # ¡Es una pérdida! Calcular la diferencia en Pips
            open_price = float(details.get("open_quote", details.get("value", last_candle["close"])))
            close_price = float(details.get("close_quote", details.get("close_strike", last_candle["close"])))
            
            # Normalizar pips
            pips_lost = abs(open_price - close_price)
            if pips_lost < 0.005 and open_price > 10:
                pips_lost = pips_lost * 100 # JPY pairs
            elif pips_lost < 0.005:
                pips_lost = pips_lost * 10000 # Normal pairs
                
            logger.error(f"[SELF-LEARNING] Pérdida detectada. Pips diff: {pips_lost:.1f}. Registrando en memoria.")
            self._save_loss_record(op, ex, pips_lost, last_candle, decision)
            
    def _save_loss_record(self, op: str, ex: int, pips: float, candle: Dict, decision: Dict):
        file_path = "trades_history.json"
        history = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    history = json.load(f)
                except Exception as e:
                    logger.warning(f"[SELF-LEARNING] No se pudo leer {file_path}: {e}")
                
        record = {
            "asset": self.asset,
            "type": self.asset_type,
            "failed_op": op,
            "failed_ex": ex,
            "pips_difference": pips,
            "rsi": candle.get("rsi", 0),
            "vol_rel": candle.get("vol_rel", 0),
            "bb_pos": (candle.get("close", 0) - candle.get("bb_lower", 0)) / (candle.get("bb_width", 1) or 1),
            "timestamp": datetime.now().isoformat(),
            "ai_reasoning": decision.get("an", "")
        }
        history.append(record)
        # Mantener solo los últimos 15 errores globales
        history = history[-15:]
        
        with open(file_path, "w") as f:
            json.dump(history, f, indent=2)

    # ─── Helpers de Notificación y Log ────────────────────────────────────────

    def _log_decision(self, decision: Dict, elapsed: float) -> None:
        """Imprime la decisión raw con parseo de seguridad."""
        op = decision.get("op", "WAIT")
        pr = decision.get("pr", 0)
        ex = decision.get("ex", 0)
        an = decision.get("an", "")
        
        arrow = "📈" if op == "CALL" else "📉" if op == "PUT" else "⏳"
        logger.info(
            f"[IA → {elapsed:.1f}s] {arrow} {op} | Confianza: {pr}% | "
            f"Expiración: {ex}m | Razón: {an}"
        )

    def _notify(self, type_: str, message: str, data: Dict = None) -> None:
        """Envía notificaciones instantáneas al websocket si hay oyentes conectados."""
        if self.on_notification:
            payload = {
                "type": type_,
                "message": message,
                "data": data or {}
            }
            try:
                # Se lanza como tarea para no bloquear el loop
                asyncio.create_task(self.on_notification(payload))
            except Exception as e:
                logger.error(f"Error al enviar notificación: {e}")

    def _add_signal(self, signal: Dict) -> None:
        """Añade una señal al historial en memoria (máximo 5)."""
        # Formatear el TS
        signal["ts"] = datetime.now().strftime("%H:%M:%S")
        self.signal_history = [signal] + self.signal_history[:4]
        # Además empujamos un evento especial HISTORY para que el frontend sincronice
        self._notify("HISTORY", "Historial actualizado", data={"history": self.signal_history})


# ─── Funciones de Log de Consola ─────────────────────────────────────────────

def _log_banner(msg: str) -> None:
    """Log de banner visual para eventos críticos del bot."""
    border = "═" * max(len(msg) + 4, 64)
    logger.info(border)
    logger.info(f"  {msg}")
    logger.info(border)


def _log_indicators(df: pd.DataFrame) -> None:
    """Log compacto de los indicadores de la vela actual."""
    last    = df.iloc[-1]
    price   = float(last["close"])
    ema200  = float(last["ema200"])
    rsi     = float(last["rsi"])
    bb_w    = float(last["bb_width"])
    vol_rel = float(last["vol_rel"])
    trend   = "↑ BULL" if price > ema200 else "↓ BEAR"

    logger.info(
        f"[INDICADORES] Precio: {price:.5f} | EMA200: {ema200:.5f} ({trend}) | "
        f"RSI: {rsi:.1f} | BB-Width: {bb_w:.3f}% | VolRel: {vol_rel:.2f}x"
    )


# ─── Instancia singleton del bot ──────────────────────────────────────────────
trading_bot = TradingBot()
