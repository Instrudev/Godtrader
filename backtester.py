"""
backtester.py – Motor de Backtesting para el bot de trading OTC

Reproduce la lógica completa del bot vela a vela sobre datos históricos.
Sin look-ahead bias: en cada vela solo tiene acceso a los datos hasta esa vela.

Uso:
    # Recolectar datos históricos del broker (requiere conexión activa):
    python backtester.py collect --asset EURUSD-OTC --candles 5000 --out data/EURUSD-OTC.csv

    # Correr backtest sobre CSV recolectado:
    python backtester.py run --csv data/EURUSD-OTC.csv --asset EURUSD-OTC

    # Backtest con rango de fechas (filtra el CSV):
    python backtester.py run --csv data/EURUSD-OTC.csv --asset EURUSD-OTC \\
                             --from 2026-03-01 --to 2026-04-01

Criterios de aceptación (Fase 5):
    Profit Factor ≥ 1.4 | Winrate ≥ 60% (payout 0.80) | Max Drawdown ≤ 15%
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from database import DB_PATH, init_db, insert_trade, load_candles, update_trade_result
from indicators import build_dataframe, pre_qualify

logger = logging.getLogger(__name__)

# ─── Configuración del Backtester ─────────────────────────────────────────────

MIN_CANDLES_FOR_SIGNAL = 205   # igual que el bot real
DEFAULT_PAYOUT         = 0.80
DEFAULT_EXPIRY_MIN     = 2
DEFAULT_AMOUNT         = 1.0
MIN_PROBABILITY        = 0.78  # igual que trader.py


# ─── Motor de Decisión Determinista (sin LLM) ─────────────────────────────────

def _infer_direction(df: pd.DataFrame) -> Optional[Tuple[str, float]]:
    """
    Aproxima la decisión del bot sin llamar a Ollama.
    Usa RSI + posición en BB para determinar dirección y confianza.

    Esta función es el "oráculo" del backtester baseline.
    En Fase 4 se reemplaza por ml_classifier.predict_proba().

    Returns:
        (direction, proba) o None si no hay señal clara.
    """
    last     = df.iloc[-1]
    rsi      = float(last["rsi"])
    price    = float(last["close"])
    bb_upper = float(last["bb_upper"])
    bb_lower = float(last["bb_lower"])
    bb_range = bb_upper - bb_lower
    pct_b    = (price - bb_lower) / bb_range if bb_range > 1e-10 else 0.5

    # EMA alineación (GATE-2 del bot real)
    ema200  = float(last["ema200"])
    ema20   = float(last["ema20"])
    macro_b = price > ema200
    micro_b = price > ema20

    # RSI sobrevendido + BB zona inferior → CALL
    if rsi < 35 and pct_b <= 0.20:
        if macro_b or micro_b:       # GATE-2: al menos una EMA alineada
            # Confianza proporcional a extremo del RSI y profundidad en banda
            confidence = 0.78 + (35 - rsi) / 35 * 0.10 + (0.20 - pct_b) * 0.10
            return "CALL", min(confidence, 0.95)

    # RSI sobrecomprado + BB zona superior → PUT
    if rsi > 65 and pct_b >= 0.80:
        if not macro_b or not micro_b:   # GATE-2 para PUT
            confidence = 0.78 + (rsi - 65) / 35 * 0.10 + (pct_b - 0.80) * 0.10
            return "PUT", min(confidence, 0.95)

    # BB Body Reversal → PUT
    # Vela verde con bb_mid en el 10% inferior del cuerpo: extensión sobre la media
    open_     = float(last["open"])
    close_    = float(last["close"])
    bb_mid    = float(last["bb_mid"])
    bb_width  = float(last["bb_width"])
    vol_rel   = float(last.get("vol_rel", 1.0))
    body      = close_ - open_

    if (body > 0
            and bb_mid <= open_ + body * 0.10
            and rsi > 55
            and bb_width > 0.30
            and vol_rel >= 1.0):
        ext   = max(0.0, (open_ + body * 0.10 - bb_mid) / body)
        confidence = 0.78 + min(ext * 0.05, 0.05) + min((rsi - 55) / 45 * 0.07, 0.07)
        return "PUT", min(confidence, 0.95)

    return None


# ─── Dataclass de Resultado ───────────────────────────────────────────────────

@dataclass
class BacktestReport:
    asset:            str
    date_from:        str
    date_to:          str
    total_candles:    int     = 0
    total_signals:    int     = 0
    total_trades:     int     = 0
    wins:             int     = 0
    losses:           int     = 0
    winrate:          float   = 0.0
    profit_factor:    float   = 0.0
    net_profit:       float   = 0.0
    max_drawdown_pct: float   = 0.0
    sharpe:           float   = 0.0
    payout:           float   = DEFAULT_PAYOUT
    amount:           float   = DEFAULT_AMOUNT
    equity_curve:     List[float]              = field(default_factory=list)
    monthly_returns:  Dict[str, float]         = field(default_factory=dict)
    hourly_winrate:   Dict[int, float]         = field(default_factory=dict)
    weekday_winrate:  Dict[int, float]         = field(default_factory=dict)

    def summary(self) -> str:
        wr = f"{self.winrate * 100:.1f}%"
        pf = f"{self.profit_factor:.2f}"
        dd = f"{self.max_drawdown_pct:.1f}%"
        sh = f"{self.sharpe:.2f}"
        meets = (
            self.profit_factor >= 1.4
            and self.winrate >= 0.60
            and self.max_drawdown_pct <= 15.0
        )
        status = "✓ CUMPLE criterios Fase 5" if meets else "✗ NO cumple criterios Fase 5"

        lines = [
            f"══ BACKTEST REPORT — {self.asset} ══",
            f"  Período:       {self.date_from} → {self.date_to}",
            f"  Velas totales: {self.total_candles}",
            f"  Señales:       {self.total_signals}",
            f"  Trades:        {self.total_trades} (wins={self.wins} losses={self.losses})",
            f"  Winrate:       {wr}",
            f"  Profit Factor: {pf}",
            f"  Net Profit:    {self.net_profit:+.2f} usd",
            f"  Max Drawdown:  {dd}",
            f"  Sharpe:        {sh}",
            f"  {status}",
        ]
        if self.hourly_winrate:
            best_h = max(self.hourly_winrate, key=self.hourly_winrate.get)
            worst_h = min(self.hourly_winrate, key=self.hourly_winrate.get)
            lines.append(
                f"  Mejor hora:    {best_h:02d}:00 UTC ({self.hourly_winrate[best_h]*100:.0f}%)"
            )
            lines.append(
                f"  Peor hora:     {worst_h:02d}:00 UTC ({self.hourly_winrate[worst_h]*100:.0f}%)"
            )
        return "\n".join(lines)


# ─── Clase Principal ──────────────────────────────────────────────────────────

class Backtester:
    """
    Reproduce la lógica del bot vela a vela sobre datos históricos.

    El motor de decisión es determinista (_infer_direction) para evitar
    llamadas a Ollama en el loop. En Fase 4 se reemplaza por ml_classifier.
    """

    def __init__(
        self,
        payout: float = DEFAULT_PAYOUT,
        amount: float = DEFAULT_AMOUNT,
        expiry_min: int = DEFAULT_EXPIRY_MIN,
        save_to_db: bool = True,
        db_path: Path = DB_PATH,
    ) -> None:
        self.payout     = payout
        self.amount     = amount
        self.expiry_min = expiry_min
        self.save_to_db = save_to_db
        self.db_path    = db_path
        if save_to_db:
            init_db(db_path)

    def run(
        self,
        candles: List[Dict],
        asset: str,
        date_from: Optional[str] = None,
        date_to:   Optional[str] = None,
    ) -> BacktestReport:
        """
        Ejecuta el backtest sobre la lista de velas proporcionada.

        Args:
            candles:   Lista de dicts OHLCV ordenada cronológicamente.
            asset:     Nombre del activo (para el reporte).
            date_from: Filtro opcional de fecha inicio (ISO-8601).
            date_to:   Filtro opcional de fecha fin (ISO-8601).

        Returns:
            BacktestReport con todas las métricas.
        """
        df_all = build_dataframe(candles)
        if df_all.empty:
            logger.error("DataFrame vacío. Se necesitan al menos 20 velas.")
            return BacktestReport(asset=asset, date_from="", date_to="")

        # Filtrar por rango de fechas si se indicó
        if date_from:
            df_all = df_all[df_all["time"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to:
            df_all = df_all[df_all["time"] <= pd.Timestamp(date_to, tz="UTC")]

        if len(df_all) < MIN_CANDLES_FOR_SIGNAL:
            logger.error(f"Datos insuficientes tras filtro: {len(df_all)} velas.")
            return BacktestReport(asset=asset, date_from="", date_to="")

        report = BacktestReport(
            asset     = asset,
            date_from = str(df_all["time"].iloc[0]),
            date_to   = str(df_all["time"].iloc[-1]),
            payout    = self.payout,
            amount    = self.amount,
        )
        report.total_candles = len(df_all)

        equity = self.amount * 100  # capital inicial arbitrario para equity curve
        peak   = equity
        equity_curve: List[float] = [equity]
        returns: List[float] = []
        trades: List[Dict] = []

        # ── Replay vela a vela ────────────────────────────────────────────────
        for i in range(MIN_CANDLES_FOR_SIGNAL, len(df_all) - self.expiry_min):
            window = df_all.iloc[: i + 1]   # solo datos hasta esta vela, sin look-ahead

            qualifies, _ = pre_qualify(window)
            if not qualifies:
                continue

            report.total_signals += 1

            result = _infer_direction(window)
            if result is None:
                continue
            direction, proba = result
            if proba < MIN_PROBABILITY:
                continue

            report.total_trades += 1

            # Resultado real: N velas adelante
            entry_candle  = df_all.iloc[i]
            close_candle  = df_all.iloc[i + self.expiry_min]
            entry_price   = float(entry_candle["close"])
            close_price   = float(close_candle["close"])

            win = (
                close_price > entry_price if direction == "CALL"
                else close_price < entry_price
            )

            profit  = self.amount * self.payout if win else -self.amount
            result_str = "WIN" if win else "LOSS"
            pips    = _pips(entry_price, close_price)

            equity += profit
            equity_curve.append(equity)
            returns.append(profit / self.amount)

            if win:
                report.wins += 1
            else:
                report.losses += 1

            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100
            report.max_drawdown_pct = max(report.max_drawdown_pct, drawdown)

            # Estadísticas horarias y por día
            ts = entry_candle["time"]
            hour    = ts.hour
            weekday = ts.weekday()

            trade_rec = {
                "ts": str(ts), "direction": direction,
                "entry": entry_price, "close": close_price,
                "profit": profit, "result": result_str,
                "hour": hour, "weekday": weekday,
            }
            trades.append(trade_rec)

            if self.save_to_db:
                self._persist(entry_candle, direction, proba, profit,
                              entry_price, close_price, pips, result_str, asset, str(ts),
                              self.db_path)

        # ── Métricas finales ──────────────────────────────────────────────────
        if report.total_trades > 0:
            report.winrate       = report.wins / report.total_trades
            gross_profit = sum(t["profit"] for t in trades if t["profit"] > 0)
            gross_loss   = abs(sum(t["profit"] for t in trades if t["profit"] < 0))
            report.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
            report.net_profit    = sum(t["profit"] for t in trades)
            report.equity_curve  = equity_curve

            if len(returns) > 1:
                std = float(np.std(returns, ddof=1))
                report.sharpe = float(np.mean(returns) / std * np.sqrt(252 * 24 * 60 / self.expiry_min)) if std > 0 else 0.0

            report.hourly_winrate  = _group_winrate(trades, "hour")
            report.weekday_winrate = _group_winrate(trades, "weekday")
            report.monthly_returns = _monthly_returns(trades)

        return report

    def run_from_db(
        self,
        asset: str,
        interval_s: int = 60,
        limit: int = 500,
        date_from: Optional[str] = None,
        date_to:   Optional[str] = None,
    ) -> BacktestReport:
        """
        Ejecuta el backtest usando las velas guardadas en candles_history (SQLite).

        Equivalente a run() pero carga los datos desde la BD en lugar de un CSV.
        Las velas se guardan automáticamente al llamar GET /candles/{asset}.

        Args:
            asset:      Nombre del activo (e.g. "EURUSD-OTC").
            interval_s: Duración de cada vela en segundos (default 60).
            limit:      Máximo de velas a cargar desde la BD (default 500).
            date_from:  Filtro opcional de fecha inicio (ISO-8601).
            date_to:    Filtro opcional de fecha fin (ISO-8601).

        Returns:
            BacktestReport con todas las métricas.

        Raises:
            ValueError: Si no hay velas guardadas para el activo en la BD.
        """
        candles = load_candles(asset, interval_s=interval_s, limit=limit, path=self.db_path)
        if not candles:
            raise ValueError(
                f"No hay velas en candles_history para '{asset}' (interval={interval_s}s). "
                "Selecciona el activo en el frontend para descargar los datos primero."
            )
        logger.info(f"run_from_db: {len(candles)} velas cargadas de BD para {asset}")
        return self.run(candles, asset, date_from=date_from, date_to=date_to)

    def _persist(
        self,
        entry_candle: pd.Series,
        direction: str,
        proba: float,
        profit: float,
        entry_price: float,
        close_price: float,
        pips: float,
        result: str,
        asset: str,
        ts: str,
        db_path: Path = DB_PATH,
    ) -> None:
        """Guarda el trade del backtest en SQLite para análisis posterior."""
        record = {
            "timestamp":       ts,
            "asset":           asset,
            "direction":       direction,
            "expiry_min":      self.expiry_min,
            "mode":            "backtest",
            "price":           float(entry_candle["close"]),
            "rsi":             float(entry_candle["rsi"]),
            "bb_pct_b":        _pct_b_from_row(entry_candle),
            "bb_width_pct":    float(entry_candle["bb_width"]),
            "vol_rel":         float(entry_candle["vol_rel"]),
            "ema20":           float(entry_candle["ema20"]),
            "ema200":          float(entry_candle["ema200"]),
            "hour_utc":        entry_candle["time"].hour,
            "weekday":         entry_candle["time"].weekday(),
            "predicted_proba": proba,
            "ai_reasoning":    "backtest-deterministic",
            "order_id":        -1,
            "open_price":      entry_price,
            "payout":          self.payout,
            "result":          result,
            "profit":          profit,
            "close_price":     close_price,
            "pips_difference": pips,
            "closed_at":       ts,
        }
        try:
            insert_trade(record, path=db_path)
        except Exception as e:
            logger.warning(f"Error guardando backtest trade en DB: {e}")


# ─── Recolección de Datos Históricos ─────────────────────────────────────────

def collect_history(
    asset: str,
    n_candles: int = 5000,
    interval: int = 60,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Obtiene datos históricos de Exnova paginando hacia atrás en el tiempo.

    Requiere que iqservice esté conectado antes de llamar esta función.

    Args:
        asset:       Nombre del activo (e.g., "EURUSD-OTC").
        n_candles:   Total de velas a recolectar (máx ~5000 recomendado).
        interval:    Segundos por vela (default 60).
        output_path: Si se indica, guarda el CSV en esta ruta.

    Returns:
        Lista de dicts OHLCV ordenada cronológicamente.
    """
    from iqservice import iq_service

    if not iq_service.is_connected():
        raise RuntimeError("iqservice no está conectado. Conectar primero con /connect.")

    PAGE_SIZE = 1000
    all_candles: List[Dict] = []
    end_time = time.time()

    pages = (n_candles + PAGE_SIZE - 1) // PAGE_SIZE
    logger.info(f"Recolectando {n_candles} velas de {asset} en {pages} páginas...")

    for page in range(pages):
        remaining = n_candles - len(all_candles)
        count = min(PAGE_SIZE, remaining)
        try:
            raw = iq_service.api.get_candles(asset, interval, count, end_time)
        except Exception as e:
            logger.error(f"Error en página {page}: {e}")
            break

        if not raw:
            logger.warning(f"Página {page} devolvió vacío. Deteniendo recolección.")
            break

        batch = [
            {
                "time":  int(c.get("from", 0)),
                "open":  float(c.get("open",  0)),
                "high":  float(c.get("max",   0)),
                "low":   float(c.get("min",   0)),
                "close": float(c.get("close", 0)),
            }
            for c in raw
        ]

        all_candles = batch + all_candles
        # Mover end_time al inicio del lote más antiguo
        end_time = batch[0]["time"] - 1
        logger.info(f"  Página {page+1}/{pages}: {len(batch)} velas | total={len(all_candles)}")
        time.sleep(0.3)  # respetar rate limit del broker

    logger.info(f"Recolección completa: {len(all_candles)} velas de {asset}")

    if output_path:
        _save_csv(all_candles, output_path)
        logger.info(f"CSV guardado en {output_path}")

    return all_candles


def load_csv(path: Path) -> List[Dict]:
    """Carga un CSV de velas previamente guardado con collect_history."""
    candles = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                "time":  int(row["time"]),
                "open":  float(row["open"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
                "close": float(row["close"]),
            })
    return candles


def _save_csv(candles: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "open", "high", "low", "close"])
        writer.writeheader()
        writer.writerows(candles)


# ─── Helpers de Métricas ──────────────────────────────────────────────────────

def _group_winrate(trades: List[Dict], key: str) -> Dict[int, float]:
    from collections import defaultdict
    wins_by: Dict[int, int] = defaultdict(int)
    total_by: Dict[int, int] = defaultdict(int)
    for t in trades:
        k = t[key]
        total_by[k] += 1
        if t["result"] == "WIN":
            wins_by[k] += 1
    return {
        k: wins_by[k] / total_by[k]
        for k in total_by
        if total_by[k] >= 3
    }


def _monthly_returns(trades: List[Dict]) -> Dict[str, float]:
    from collections import defaultdict
    by_month: Dict[str, float] = defaultdict(float)
    for t in trades:
        month = t["ts"][:7]  # "YYYY-MM"
        by_month[month] += t["profit"]
    return dict(by_month)


def _pips(a: float, b: float) -> float:
    """Convierte diferencia absoluta en pips (heurística OTC forex).
    JPY y pares con precio alto: 1 pip = 0.01 → multiplicar por 100.
    Pares normales (EURUSD etc.): 1 pip = 0.0001 → multiplicar por 10000.
    """
    diff = abs(a - b)
    if a > 10:          # JPY, índices OTC (precio alto)
        return diff * 100
    return diff * 10000  # pares normales


def _pct_b_from_row(row: pd.Series) -> float:
    price    = float(row["close"])
    bb_upper = float(row["bb_upper"])
    bb_lower = float(row["bb_lower"])
    bb_range = bb_upper - bb_lower
    if bb_range <= 1e-10:
        return 0.5
    return (price - bb_lower) / bb_range


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Backtester del bot de trading OTC")
    sub = parser.add_subparsers(dest="cmd")

    # Subcomando: collect
    col = sub.add_parser("collect", help="Recolectar datos históricos del broker")
    col.add_argument("--asset",   required=True, help="Nombre del activo (e.g. EURUSD-OTC)")
    col.add_argument("--candles", type=int, default=5000, help="Número de velas a recolectar")
    col.add_argument("--out",     default=None, help="Ruta del CSV de salida")

    # Subcomando: run-db
    rdb = sub.add_parser("run-db", help="Ejecutar backtest desde candles_history en SQLite")
    rdb.add_argument("--asset",    required=True, help="Nombre del activo (e.g. EURUSD-OTC)")
    rdb.add_argument("--interval", type=int, default=60,             help="Segundos por vela (default 60)")
    rdb.add_argument("--limit",    type=int, default=500,            help="Máx velas a cargar (default 500)")
    rdb.add_argument("--from",     dest="date_from", default=None,   help="Fecha inicio YYYY-MM-DD")
    rdb.add_argument("--to",       dest="date_to",   default=None,   help="Fecha fin YYYY-MM-DD")
    rdb.add_argument("--payout",   type=float, default=DEFAULT_PAYOUT, help="Payout asumido")
    rdb.add_argument("--amount",   type=float, default=DEFAULT_AMOUNT,  help="Monto por trade")
    rdb.add_argument("--no-db",    action="store_true", help="No guardar trades en SQLite")

    # Subcomando: run
    run = sub.add_parser("run", help="Ejecutar backtest sobre CSV")
    run.add_argument("--csv",    required=True, help="CSV con datos históricos")
    run.add_argument("--asset",  required=True, help="Nombre del activo")
    run.add_argument("--from",   dest="date_from", default=None, help="Fecha inicio YYYY-MM-DD")
    run.add_argument("--to",     dest="date_to",   default=None, help="Fecha fin YYYY-MM-DD")
    run.add_argument("--payout", type=float, default=DEFAULT_PAYOUT, help="Payout asumido")
    run.add_argument("--amount", type=float, default=DEFAULT_AMOUNT,  help="Monto por trade")
    run.add_argument("--no-db",  action="store_true", help="No guardar trades en SQLite")

    args = parser.parse_args()

    if args.cmd == "run-db":
        bt = Backtester(
            payout=args.payout,
            amount=args.amount,
            save_to_db=not args.no_db,
        )
        try:
            report = bt.run_from_db(
                args.asset, args.interval, args.limit, args.date_from, args.date_to
            )
        except ValueError as e:
            print(f"Error: {e}")
            return
        print(report.summary())

        report_path = Path(f"backtest_{args.asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        data = {
            "asset":            report.asset,
            "date_from":        report.date_from,
            "date_to":          report.date_to,
            "total_candles":    report.total_candles,
            "total_trades":     report.total_trades,
            "wins":             report.wins,
            "losses":           report.losses,
            "winrate":          report.winrate,
            "profit_factor":    report.profit_factor,
            "net_profit":       report.net_profit,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe":           report.sharpe,
            "monthly_returns":  report.monthly_returns,
            "hourly_winrate":   {str(k): v for k, v in report.hourly_winrate.items()},
            "weekday_winrate":  {str(k): v for k, v in report.weekday_winrate.items()},
        }
        report_path.write_text(json.dumps(data, indent=2))
        print(f"\nReporte JSON guardado en: {report_path}")

    elif args.cmd == "collect":
        # Necesita conexión al broker; esto se usa desde la shell con el servidor activo
        print("Para recolectar datos históricos, usa el endpoint /candles/{asset} de la API")
        print("o conéctate manualmente e importa collect_history desde este módulo.")

    elif args.cmd == "run":
        candles = load_csv(Path(args.csv))
        bt = Backtester(
            payout=args.payout,
            amount=args.amount,
            save_to_db=not args.no_db,
        )
        report = bt.run(candles, args.asset, args.date_from, args.date_to)
        print(report.summary())

        # Guardar JSON con reporte completo
        report_path = Path(f"backtest_{args.asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        data = {
            "asset":            report.asset,
            "date_from":        report.date_from,
            "date_to":          report.date_to,
            "total_candles":    report.total_candles,
            "total_trades":     report.total_trades,
            "wins":             report.wins,
            "losses":           report.losses,
            "winrate":          report.winrate,
            "profit_factor":    report.profit_factor,
            "net_profit":       report.net_profit,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe":           report.sharpe,
            "monthly_returns":  report.monthly_returns,
            "hourly_winrate":   {str(k): v for k, v in report.hourly_winrate.items()},
            "weekday_winrate":  {str(k): v for k, v in report.weekday_winrate.items()},
        }
        report_path.write_text(json.dumps(data, indent=2))
        print(f"\nReporte JSON guardado en: {report_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
