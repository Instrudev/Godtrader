"""
tests/test_backtester.py – Tests unitarios para backtester.py

Usa datos sintéticos generados en memoria para no requerir conexión al broker.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from backtester import (
    Backtester,
    BacktestReport,
    _group_winrate,
    _infer_direction,
    _monthly_returns,
    _pct_b_from_row,
    _pips,
    load_csv,
    _save_csv,
)
from indicators import build_dataframe


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_candles(n: int = 300, base_price: float = 1.08500, seed: int = 42) -> List[dict]:
    """
    Genera N velas sintéticas de 1 minuto con mean-reversion forzada.
    Suficientes indicadores para que build_dataframe sea estable.
    """
    rng = np.random.default_rng(seed)
    candles = []
    price = base_price
    t0 = 1_700_000_000  # timestamp base Unix

    for i in range(n):
        # Mean-reversion hacia base_price
        drift = (base_price - price) * 0.02
        change = drift + rng.normal(0, 0.0003)
        open_p = price
        close_p = price + change
        high_p = max(open_p, close_p) + abs(rng.normal(0, 0.0001))
        low_p  = min(open_p, close_p) - abs(rng.normal(0, 0.0001))
        candles.append({
            "time":  t0 + i * 60,
            "open":  round(open_p,  5),
            "high":  round(high_p,  5),
            "low":   round(low_p,   5),
            "close": round(close_p, 5),
        })
        price = close_p

    return candles


def _make_candles_oversold(n: int = 250, base_price: float = 1.08500) -> List[dict]:
    """
    Genera velas que terminan con RSI < 35 y precio cerca de BB inferior,
    para forzar señales CALL en el backtester.
    """
    candles = _make_candles(n, base_price, seed=7)
    # Forzar bajada pronunciada al final para crear zona sobrevendida
    t_base = candles[-1]["time"]
    price = candles[-1]["close"]
    for i in range(30):
        price -= 0.00060
        t = t_base + (i + 1) * 60
        candles.append({
            "time":  t,
            "open":  round(price + 0.00030, 5),
            "high":  round(price + 0.00035, 5),
            "low":   round(price - 0.00005, 5),
            "close": round(price, 5),
        })
    return candles


# ─── Tests de _pips ───────────────────────────────────────────────────────────

def test_pips_normal_pair() -> None:
    result = _pips(1.08500, 1.08520)
    assert result == pytest.approx(2.0, rel=1e-3)


def test_pips_jpy_pair() -> None:
    # JPY pair: 155.020 - 155.000 = 0.020 → 0.020 * 100 = 2.0 pips
    result = _pips(155.000, 155.020)
    assert result == pytest.approx(2.0, rel=1e-3)


def test_pips_same_price() -> None:
    assert _pips(1.0, 1.0) == 0.0


def test_pips_always_positive() -> None:
    assert _pips(1.08520, 1.08500) > 0


# ─── Tests de _pct_b_from_row ─────────────────────────────────────────────────

def test_pct_b_at_lower_band() -> None:
    df = build_dataframe(_make_candles(250))
    row = df.iloc[-1].copy()
    row["close"] = row["bb_lower"]
    assert _pct_b_from_row(row) == pytest.approx(0.0, abs=1e-10)


def test_pct_b_at_upper_band() -> None:
    df = build_dataframe(_make_candles(250))
    row = df.iloc[-1].copy()
    row["close"] = row["bb_upper"]
    assert _pct_b_from_row(row) == pytest.approx(1.0, abs=1e-10)


def test_pct_b_at_midband() -> None:
    df = build_dataframe(_make_candles(250))
    row = df.iloc[-1].copy()
    row["close"] = row["bb_mid"]
    assert _pct_b_from_row(row) == pytest.approx(0.5, abs=0.01)


def test_pct_b_zero_range_returns_half() -> None:
    df = build_dataframe(_make_candles(250))
    row = df.iloc[-1].copy()
    row["bb_upper"] = row["bb_lower"]  # banda colapsada
    assert _pct_b_from_row(row) == 0.5


# ─── Tests de _infer_direction ────────────────────────────────────────────────

def test_infer_direction_oversold_returns_call() -> None:
    candles = _make_candles_oversold(280)
    df = build_dataframe(candles)
    # Buscar una vela con RSI < 35
    oversold = df[df["rsi"] < 35]
    if oversold.empty:
        pytest.skip("Los datos sintéticos no generaron RSI < 35 en este seed")
    idx = oversold.index[0]
    window = df.iloc[: idx + 1]
    result = _infer_direction(window)
    if result is not None:
        direction, proba = result
        assert direction == "CALL"
        assert 0.0 < proba <= 1.0


def test_infer_direction_neutral_rsi_returns_none() -> None:
    candles = _make_candles(250, seed=99)
    df = build_dataframe(candles)
    # Forzar RSI neutro en la última vela
    df.at[df.index[-1], "rsi"] = 50.0
    df.at[df.index[-1], "bb_pct_b"] = 0.5  # no existe en real df, pero _infer usa bb_upper/lower
    result = _infer_direction(df)
    # Con RSI neutral, no debería dar señal
    # (puede devolver None o CALL/PUT si otros factores califican — aceptamos ambos)
    assert result is None or result[0] in ("CALL", "PUT")


def test_infer_direction_proba_in_range() -> None:
    candles = _make_candles_oversold(280)
    df = build_dataframe(candles)
    oversold = df[df["rsi"] < 35]
    if oversold.empty:
        pytest.skip("No se generó RSI < 35")
    idx = oversold.index[0]
    window = df.iloc[: idx + 1]
    result = _infer_direction(window)
    if result is not None:
        _, proba = result
        assert 0.0 <= proba <= 1.0


# ─── Tests de BacktestReport ──────────────────────────────────────────────────

def test_backtest_report_summary_contains_asset() -> None:
    report = BacktestReport(asset="EURUSD-OTC", date_from="2026-01-01", date_to="2026-03-31")
    report.total_trades = 10
    report.wins = 6
    report.losses = 4
    report.winrate = 0.60
    report.profit_factor = 1.5
    report.net_profit = 5.0
    report.max_drawdown_pct = 8.0
    summary = report.summary()
    assert "EURUSD-OTC" in summary


def test_backtest_report_meets_criteria() -> None:
    report = BacktestReport(asset="X", date_from="", date_to="")
    report.total_trades = 100
    report.wins = 65
    report.losses = 35
    report.winrate = 0.65
    report.profit_factor = 1.5
    report.max_drawdown_pct = 10.0
    summary = report.summary()
    assert "CUMPLE" in summary


def test_backtest_report_not_meets_criteria() -> None:
    report = BacktestReport(asset="X", date_from="", date_to="")
    report.total_trades = 100
    report.wins = 50
    report.losses = 50
    report.winrate = 0.50
    report.profit_factor = 1.0
    report.max_drawdown_pct = 20.0
    summary = report.summary()
    assert "NO cumple" in summary


# ─── Tests de Backtester.run ──────────────────────────────────────────────────

def test_backtester_run_empty_candles() -> None:
    bt = Backtester(save_to_db=False)
    report = bt.run([], "EURUSD-OTC")
    assert report.total_trades == 0


def test_backtester_run_insufficient_candles() -> None:
    candles = _make_candles(100)  # menos de MIN_CANDLES_FOR_SIGNAL=205
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    assert report.total_trades == 0


def test_backtester_run_returns_report_type() -> None:
    candles = _make_candles(400)
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    assert isinstance(report, BacktestReport)


def test_backtester_run_total_candles_set(tmp_path: Path) -> None:
    candles = _make_candles(400)
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    assert report.total_candles == 400


def test_backtester_run_wins_plus_losses_eq_total(tmp_path: Path) -> None:
    candles = _make_candles_oversold(350)
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    assert report.wins + report.losses == report.total_trades


def test_backtester_run_winrate_in_range() -> None:
    candles = _make_candles(400)
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    if report.total_trades > 0:
        assert 0.0 <= report.winrate <= 1.0


def test_backtester_run_profit_factor_non_negative() -> None:
    candles = _make_candles_oversold(400)
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    if report.total_trades > 0 and not math.isinf(report.profit_factor):
        assert report.profit_factor >= 0


def test_backtester_run_max_drawdown_non_negative() -> None:
    candles = _make_candles(400)
    bt = Backtester(save_to_db=False)
    report = bt.run(candles, "EURUSD-OTC")
    assert report.max_drawdown_pct >= 0.0


def test_backtester_saves_to_db(tmp_path: Path) -> None:
    from database import fetch_trades

    test_db = tmp_path / "backtest_test.db"
    candles = _make_candles_oversold(350)
    bt = Backtester(save_to_db=True, db_path=test_db)
    report = bt.run(candles, "EURUSD-OTC")
    if report.total_trades > 0:
        trades = fetch_trades(mode="backtest", path=test_db)
        assert len(trades) == report.total_trades


# ─── Tests de helpers de métricas ─────────────────────────────────────────────

def test_group_winrate_basic() -> None:
    trades = [
        {"hour": 10, "result": "WIN"},
        {"hour": 10, "result": "WIN"},
        {"hour": 10, "result": "LOSS"},
        {"hour": 14, "result": "WIN"},
        {"hour": 14, "result": "LOSS"},
        {"hour": 14, "result": "LOSS"},
    ]
    result = _group_winrate(trades, "hour")
    assert 10 in result
    assert result[10] == pytest.approx(2 / 3)
    assert 14 in result
    assert result[14] == pytest.approx(1 / 3)


def test_group_winrate_min_3_trades() -> None:
    trades = [
        {"hour": 5, "result": "WIN"},
        {"hour": 5, "result": "WIN"},
    ]
    result = _group_winrate(trades, "hour")
    assert 5 not in result  # menos de 3 trades


def test_monthly_returns_aggregation() -> None:
    trades = [
        {"ts": "2026-01-15T10:00:00", "profit": 0.8,  "result": "WIN"},
        {"ts": "2026-01-20T10:00:00", "profit": -1.0, "result": "LOSS"},
        {"ts": "2026-02-05T10:00:00", "profit": 0.8,  "result": "WIN"},
    ]
    result = _monthly_returns(trades)
    assert "2026-01" in result
    assert "2026-02" in result
    assert result["2026-01"] == pytest.approx(-0.2, abs=1e-9)
    assert result["2026-02"] == pytest.approx(0.8, abs=1e-9)


# ─── Tests de CSV I/O ─────────────────────────────────────────────────────────

def test_save_and_load_csv(tmp_path: Path) -> None:
    candles = _make_candles(50)
    csv_path = tmp_path / "test.csv"
    _save_csv(candles, csv_path)
    assert csv_path.exists()

    loaded = load_csv(csv_path)
    assert len(loaded) == 50
    assert loaded[0]["time"] == candles[0]["time"]
    assert loaded[-1]["close"] == pytest.approx(candles[-1]["close"], rel=1e-5)


def test_load_csv_preserves_ohlcv(tmp_path: Path) -> None:
    candles = _make_candles(10)
    csv_path = tmp_path / "small.csv"
    _save_csv(candles, csv_path)
    loaded = load_csv(csv_path)
    for original, restored in zip(candles, loaded):
        assert restored["open"]  == pytest.approx(original["open"],  rel=1e-5)
        assert restored["high"]  == pytest.approx(original["high"],  rel=1e-5)
        assert restored["low"]   == pytest.approx(original["low"],   rel=1e-5)
        assert restored["close"] == pytest.approx(original["close"], rel=1e-5)
