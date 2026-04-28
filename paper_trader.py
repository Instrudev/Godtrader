"""
paper_trader.py – Modo Paper Trading (Forward Test sobre cuenta demo)

Opera exactamente como TradingBot pero:
  - Nunca envía órdenes reales al broker (simula aceptación inmediata)
  - Determina WIN/LOSS comparando precio de entrada vs precio al vencimiento
    usando el feed de precios en tiempo real de Exnova
  - Registra cada trade completo (features + resultado) en trades.db
  - Acumula el dataset de entrenamiento para el clasificador ML (Fase 4)

Uso:
    python main.py --mode paper
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple

from database import DB_PATH, init_db, insert_trade, update_trade_result
from iqservice import iq_service
from trader import CANDLE_INTERVAL, TradingBot

logger = logging.getLogger(__name__)

# Payout asumido hasta que Fase 2 implemente consulta real al broker
DEFAULT_PAYOUT = 0.80


class PaperTrader(TradingBot):
    """
    Hereda toda la lógica de TradingBot (bucle, pre-cal, IA, gatillos).
    Sobreescribe únicamente _place_order y _monitor_trade para simular.
    """

    def __init__(self, payout: float = DEFAULT_PAYOUT) -> None:
        super().__init__()
        self.payout = payout
        self._fake_order_counter: int = 0
        # Mapeo fake_order_id → db_trade_id para actualizar el resultado después
        self._db_trade_ids: Dict[int, int] = {}
        init_db(DB_PATH)
        logger.info(f"PaperTrader listo | payout asumido={payout:.2f} | DB={DB_PATH}")

    # ─── Sobreescritura de _place_order ──────────────────────────────────────

    def _place_order(self, direction: str, expiry_minutes: int) -> Tuple[bool, int]:
        """
        Simula la aceptación inmediata de una orden.
        No envía nada al broker.

        Returns:
            (True, fake_order_id) siempre.
        """
        self._fake_order_counter += 1
        fake_id = self._fake_order_counter
        logger.info(
            f"[PAPER] Orden simulada #{fake_id} | {direction} {expiry_minutes}m | {self.asset}"
        )
        return True, fake_id

    # ─── Sobreescritura de _monitor_trade ────────────────────────────────────

    async def _monitor_trade(
        self,
        order_id: int,
        op: str,
        ex: int,
        decision: Dict,
        last_candle: Dict,
    ) -> None:
        """
        1. Guarda la entrada en SQLite con todas las features.
        2. Espera al vencimiento (ex × 60s).
        3. Obtiene el precio de cierre real del feed de Exnova.
        4. Calcula WIN/LOSS y actualiza el registro en DB.
        """
        entry_price = float(last_candle.get("close", 0.0))
        entry_ts = datetime.now(timezone.utc).isoformat()
        now_utc = datetime.now(timezone.utc)

        # ── Guardar entrada ───────────────────────────────────────────────────
        ml_feats = decision.get("ml_features", {})
        payout_used = self._last_payout if hasattr(self, "_last_payout") else self.payout

        record: Dict = {
            "timestamp":       entry_ts,
            "asset":           self.asset,
            "direction":       op,
            "expiry_min":      ex,
            "mode":            "paper",
            "price":           entry_price,
            "rsi":             float(last_candle.get("rsi", 0.0)),
            "bb_pct_b":        _bb_pct_b(last_candle),
            "bb_width_pct":    float(last_candle.get("bb_width", 0.0)),
            "vol_rel":         float(last_candle.get("vol_rel", 1.0)),
            "ema20":           float(last_candle.get("ema20", 0.0)),
            "ema200":          float(last_candle.get("ema200", 0.0)),
            "hour_utc":        now_utc.hour,
            "weekday":         now_utc.weekday(),
            "predicted_proba": decision.get("pr", 0) / 100.0,
            "ai_reasoning":    decision.get("an", ""),
            "order_id":        order_id,
            "open_price":      entry_price,
            "payout":          payout_used,
            # ── Features ML (Fase 4) ──────────────────────────────────────────
            "rel_atr":         ml_feats.get("rel_atr"),
            "streak_length":   int(ml_feats["streak_length"]) if ml_feats.get("streak_length") is not None else None,
            "streak_pct":      ml_feats.get("streak_pct"),
            "ret_3":           ml_feats.get("ret_3"),
            "ret_5":           ml_feats.get("ret_5"),
            "ret_10":          ml_feats.get("ret_10"),
            "autocorr_10":     ml_feats.get("autocorr_10"),
            "half_life":       ml_feats.get("half_life"),
            # ── Features PRNG ─────────────────────────────────────────────────
            "prng_last_digit_entropy":   ml_feats.get("prng_last_digit_entropy"),
            "prng_last_digit_mode_freq": ml_feats.get("prng_last_digit_mode_freq"),
            "prng_permutation_entropy":  ml_feats.get("prng_permutation_entropy"),
            "prng_runs_test_z":          ml_feats.get("prng_runs_test_z"),
            "prng_transition_entropy":   ml_feats.get("prng_transition_entropy"),
            "prng_hurst_exponent":       ml_feats.get("prng_hurst_exponent"),
            "prng_turning_point_ratio":  ml_feats.get("prng_turning_point_ratio"),
            "prng_autocorr_lag2":        ml_feats.get("prng_autocorr_lag2"),
            "prng_autocorr_lag5":        ml_feats.get("prng_autocorr_lag5"),
        }
        db_id = insert_trade(record)
        self._db_trade_ids[order_id] = db_id
        logger.info(f"[PAPER] Entrada guardada en DB id={db_id}")

        # ── Esperar al vencimiento ────────────────────────────────────────────
        wait_secs = ex * CANDLE_INTERVAL
        logger.info(f"[PAPER] Esperando {wait_secs}s para determinar resultado...")
        await asyncio.sleep(wait_secs)

        # ── Obtener precio de cierre del feed en tiempo real ──────────────────
        close_price = entry_price  # fallback si el feed falla
        try:
            candle = await asyncio.to_thread(
                iq_service.get_realtime_candle, self.asset, CANDLE_INTERVAL
            )
            if candle and candle.get("close"):
                close_price = float(candle["close"])
        except Exception as e:
            logger.warning(f"[PAPER] Error obteniendo precio de cierre: {e}. Usando precio de entrada.")

        # ── Determinar resultado ──────────────────────────────────────────────
        if op == "CALL":
            win = close_price > entry_price
        else:  # PUT
            win = close_price < entry_price

        result = "WIN" if win else "LOSS"
        profit = self.amount * self.payout if win else -self.amount
        pips = _to_pips(entry_price, close_price)
        closed_at = datetime.now(timezone.utc).isoformat()

        # ── Actualizar DB ─────────────────────────────────────────────────────
        update_trade_result(
            trade_id=db_id,
            result=result,
            profit=profit,
            payout=self.payout,
            open_price=entry_price,
            close_price=close_price,
            pips_difference=pips,
            closed_at=closed_at,
        )

        # ── Actualizar trade_log en RAM (para /bot/status) ────────────────────
        for trade in reversed(self.trade_log):
            if trade.get("result") == "PENDING" and trade.get("op") == op:
                trade["result"] = result
                trade["profit"] = profit
                break

        arrow = "WIN ✓" if win else "LOSS ✗"
        logger.info(
            f"[PAPER] {arrow} | {op} | Entrada={entry_price:.5f} | "
            f"Cierre={close_price:.5f} | Profit={profit:+.2f} | DB id={db_id}"
        )
        self._notify(
            "EXECUTION",
            f"[PAPER] {result}: {op} en {self.asset}",
            data={"result": result, "profit": profit, "payout": self.payout, "db_id": db_id},
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bb_pct_b(candle: Dict) -> float:
    """
    Calcula pct_b correctamente: (price - bb_lower) / (bb_upper - bb_lower).
    El campo bb_width en el DataFrame es el porcentaje, NO el rango absoluto;
    por eso aquí usamos bb_upper y bb_lower directamente.
    """
    price    = float(candle.get("close",    0.0))
    bb_upper = float(candle.get("bb_upper", price))
    bb_lower = float(candle.get("bb_lower", price))
    bb_range = bb_upper - bb_lower
    if bb_range <= 1e-10:
        return 0.5
    return (price - bb_lower) / bb_range


def _to_pips(price_a: float, price_b: float) -> float:
    """Convierte diferencia absoluta de precio a pips (heurística OTC forex).
    JPY y pares con precio alto: 1 pip = 0.01 → diff * 100.
    Pares normales (EURUSD etc.): 1 pip = 0.0001 → diff * 10000.
    """
    diff = abs(price_a - price_b)
    if price_a > 10:
        return diff * 100
    return diff * 10000


# ─── Singleton ───────────────────────────────────────────────────────────────
paper_trading_bot = PaperTrader()
