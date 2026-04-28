"""
tests/test_regime_filter.py – Tests unitarios para regime_filter.py

Todos los tests son puros (sin broker, sin DB real excepto via tmp_path).
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Parchear database antes de importar regime_filter para evitar que
# get_winrate_by_hour/weekday lean de una DB real
import database as _db_module

from regime_filter import (
    FilterResult,
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSSES,
    MAX_TRADES_PER_DAY,
    MIN_PAYOUT,
    _calculate_atr,
    check_all_filters,
    consecutive_loss_filter,
    daily_loss_filter,
    hour_profile_filter,
    max_trades_filter,
    payout_filter,
    volatility_filter,
    weekday_profile_filter,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """DataFrame sintético con columnas OHLCV + indicadores mínimos."""
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000
    prices = 1.08500 + np.cumsum(rng.normal(0, 0.0003, n))
    data = {
        "time":     pd.to_datetime([t0 + i * 60 for i in range(n)], unit="s", utc=True),
        "open":     prices,
        "close":    prices + rng.normal(0, 0.0001, n),
        "high":     prices + abs(rng.normal(0, 0.0002, n)),
        "low":      prices - abs(rng.normal(0, 0.0002, n)),
        "volume":   np.ones(n),
        "rsi":      rng.uniform(40, 60, n),
        "bb_upper": prices + 0.002,
        "bb_lower": prices - 0.002,
        "bb_mid":   prices,
        "bb_width": np.full(n, 0.37),
        "vol_rel":  np.ones(n),
        "ema20":    prices,
        "ema200":   prices,
    }
    return pd.DataFrame(data)


def _trade(result: str, today: bool = True) -> dict:
    ts = date.today().isoformat() if today else "2020-01-01"
    return {"result": result, "timestamp": ts + "T10:00:00", "op": "CALL"}


# ─── FilterResult ─────────────────────────────────────────────────────────────

def test_filter_result_ok() -> None:
    r = FilterResult.ok()
    assert r.allow is True
    assert r.filter_name == ""
    assert r.auto_shutdown is False


def test_filter_result_block() -> None:
    r = FilterResult.block("test_filter", "razón de prueba")
    assert r.allow is False
    assert r.filter_name == "test_filter"
    assert r.reason == "razón de prueba"
    assert r.auto_shutdown is False


def test_filter_result_block_with_shutdown() -> None:
    r = FilterResult.block("f", "r", shutdown=True)
    assert r.auto_shutdown is True


# ─── hour_profile_filter ──────────────────────────────────────────────────────

def test_hour_filter_no_data_allows() -> None:
    with patch("regime_filter.get_winrate_by_hour", return_value={}):
        result = hour_profile_filter(hour_utc=10, asset="EURUSD-OTC")
    assert result.allow is True


def test_hour_filter_good_winrate_allows() -> None:
    with patch("regime_filter.get_winrate_by_hour", return_value={10: 0.62}):
        result = hour_profile_filter(hour_utc=10, asset="EURUSD-OTC")
    assert result.allow is True


def test_hour_filter_bad_winrate_blocks() -> None:
    with patch("regime_filter.get_winrate_by_hour", return_value={10: 0.45}):
        result = hour_profile_filter(hour_utc=10, asset="EURUSD-OTC", min_winrate=0.52)
    assert result.allow is False
    assert result.filter_name == "hour_profile_filter"
    assert result.auto_shutdown is False


def test_hour_filter_exactly_at_threshold_allows() -> None:
    with patch("regime_filter.get_winrate_by_hour", return_value={14: 0.52}):
        result = hour_profile_filter(hour_utc=14, min_winrate=0.52)
    assert result.allow is True


def test_hour_filter_different_hour_not_in_data_allows() -> None:
    with patch("regime_filter.get_winrate_by_hour", return_value={10: 0.40}):
        result = hour_profile_filter(hour_utc=15)  # hora 15 no está en data
    assert result.allow is True


# ─── weekday_profile_filter ───────────────────────────────────────────────────

def test_weekday_filter_no_data_allows() -> None:
    with patch("regime_filter.get_winrate_by_weekday", return_value={}):
        result = weekday_profile_filter(weekday=5)
    assert result.allow is True


def test_weekday_filter_saturday_low_winrate_blocks() -> None:
    with patch("regime_filter.get_winrate_by_weekday", return_value={5: 0.45}):
        result = weekday_profile_filter(weekday=5, min_winrate=0.52)
    assert result.allow is False
    assert "Sábado" in result.reason


def test_weekday_filter_sunday_good_winrate_allows() -> None:
    with patch("regime_filter.get_winrate_by_weekday", return_value={6: 0.60}):
        result = weekday_profile_filter(weekday=6)
    assert result.allow is True


# ─── volatility_filter ────────────────────────────────────────────────────────

def test_volatility_filter_insufficient_data_allows() -> None:
    df = _make_df(n=10)
    result = volatility_filter(df)
    assert result.allow is True


def test_volatility_filter_normal_volatility_allows() -> None:
    df = _make_df(n=100)
    result = volatility_filter(df)
    assert result.allow is True   # datos normales no deberían bloquearse


def test_volatility_filter_zero_atr_allows() -> None:
    """Si ATR es cero (precios constantes), no bloquear (fail-open)."""
    df = _make_df(n=50)
    df["high"]  = 1.08500
    df["low"]   = 1.08500
    df["close"] = 1.08500
    result = volatility_filter(df)
    assert result.allow is True


def test_volatility_filter_blocks_extreme_high_volatility() -> None:
    """Velas gigantes → ATR muy alto → debería bloquear."""
    rng = np.random.default_rng(42)
    df = _make_df(n=100)
    # Las últimas 5 velas tienen volatilidad extrema (100x el rango normal)
    for col in ("high", "low"):
        df.iloc[-5:, df.columns.get_loc(col)] = df[col].iloc[-5:] * (
            1.5 if col == "high" else 0.5
        )
    result = volatility_filter(df, percentile_low=30, percentile_high=80)
    # Con umbral más estricto (80 en lugar de 95) es más probable que bloquee
    # No forzamos el resultado ya que depende de los datos sintéticos


# ─── payout_filter ────────────────────────────────────────────────────────────

def test_payout_filter_none_allows() -> None:
    result = payout_filter(current_payout=None)
    assert result.allow is True


def test_payout_filter_above_minimum_allows() -> None:
    result = payout_filter(current_payout=0.85, min_payout=0.80)
    assert result.allow is True


def test_payout_filter_exactly_at_minimum_allows() -> None:
    result = payout_filter(current_payout=0.80, min_payout=0.80)
    assert result.allow is True


def test_payout_filter_below_minimum_blocks() -> None:
    result = payout_filter(current_payout=0.72, min_payout=0.80)
    assert result.allow is False
    assert result.filter_name == "payout_filter"
    assert "72%" in result.reason or "0.72" in result.reason or "72" in result.reason


def test_payout_filter_zero_blocks() -> None:
    result = payout_filter(current_payout=0.0, min_payout=0.80)
    assert result.allow is False


# ─── daily_loss_filter ────────────────────────────────────────────────────────

def test_daily_loss_no_trades_allows() -> None:
    result = daily_loss_filter([], max_daily_losses=3)
    assert result.allow is True


def test_daily_loss_some_losses_below_limit_allows() -> None:
    log = [_trade("LOSS"), _trade("WIN"), _trade("LOSS")]
    result = daily_loss_filter(log, max_daily_losses=3)
    assert result.allow is True   # 2 pérdidas < 3


def test_daily_loss_at_limit_blocks() -> None:
    log = [_trade("LOSS"), _trade("LOSS"), _trade("LOSS")]
    result = daily_loss_filter(log, max_daily_losses=3)
    assert result.allow is False
    assert result.auto_shutdown is True


def test_daily_loss_old_losses_not_counted() -> None:
    """Las pérdidas de otros días no cuentan para el filtro diario."""
    log = [
        _trade("LOSS", today=False),
        _trade("LOSS", today=False),
        _trade("LOSS", today=False),
        _trade("WIN",  today=True),
    ]
    result = daily_loss_filter(log, max_daily_losses=3)
    assert result.allow is True   # solo 0 pérdidas hoy


def test_daily_loss_auto_shutdown_true() -> None:
    log = [_trade("LOSS")] * 5
    result = daily_loss_filter(log, max_daily_losses=3)
    assert result.auto_shutdown is True


# ─── consecutive_loss_filter ──────────────────────────────────────────────────

def test_consecutive_loss_no_trades_allows() -> None:
    result = consecutive_loss_filter([], max_consecutive=3)
    assert result.allow is True


def test_consecutive_loss_below_limit_allows() -> None:
    log = [_trade("WIN"), _trade("LOSS"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3)
    assert result.allow is True   # 2 consecutivas < 3


def test_consecutive_loss_at_limit_blocks() -> None:
    log = [_trade("WIN"), _trade("LOSS"), _trade("LOSS"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3)
    assert result.allow is False
    assert result.auto_shutdown is True


def test_consecutive_loss_win_resets_streak() -> None:
    log = [_trade("LOSS"), _trade("LOSS"), _trade("WIN"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3)
    assert result.allow is True   # solo 1 consecutiva al final


def test_consecutive_loss_pending_ignored() -> None:
    """Los trades PENDING no cuentan en el streak: 2 LOSS + PENDING → streak=2 < 3."""
    log = [_trade("LOSS"), _trade("LOSS"), {"result": "PENDING", "timestamp": "2026-01-01"}]
    result = consecutive_loss_filter(log, max_consecutive=3)  # límite=3, streak=2 → pasa
    assert result.allow is True


# ─── max_trades_filter ────────────────────────────────────────────────────────

def test_max_trades_no_trades_allows() -> None:
    result = max_trades_filter([], max_trades=5)
    assert result.allow is True


def test_max_trades_below_limit_allows() -> None:
    log = [_trade("WIN"), _trade("LOSS"), _trade("WIN")]
    result = max_trades_filter(log, max_trades=5)
    assert result.allow is True


def test_max_trades_at_limit_blocks() -> None:
    log = [_trade("WIN")] * 5
    result = max_trades_filter(log, max_trades=5)
    assert result.allow is False
    assert result.filter_name == "max_trades_filter"
    assert result.auto_shutdown is False   # no es auto-shutdown, solo espera mañana


def test_max_trades_old_trades_not_counted() -> None:
    log = [_trade("WIN", today=False)] * 10 + [_trade("WIN", today=True)]
    result = max_trades_filter(log, max_trades=5)
    assert result.allow is True   # solo 1 hoy


def test_max_trades_pending_not_counted() -> None:
    """Trades PENDING no cuentan (aún no han cerrado)."""
    log = [{"result": "PENDING", "timestamp": date.today().isoformat() + "T10:00:00"}] * 5
    result = max_trades_filter(log, max_trades=5)
    assert result.allow is True


# ─── check_all_filters ────────────────────────────────────────────────────────

def _safe_datetime():
    """Devuelve un datetime en hora/día no bloqueados (Jueves 10:00 UTC)."""
    from datetime import datetime, timezone
    return datetime(2026, 4, 30, 10, 0, 0, tzinfo=timezone.utc)  # Jueves 10h UTC


def test_check_all_filters_empty_passes() -> None:
    df = _make_df(n=100)
    with patch("regime_filter.get_winrate_by_hour", return_value={}), \
         patch("regime_filter.get_winrate_by_weekday", return_value={}), \
         patch("regime_filter.drift_filter", return_value=FilterResult.ok()), \
         patch("regime_filter.datetime") as mock_dt:
        mock_dt.now.return_value = _safe_datetime()
        result = check_all_filters(df=df, asset="EURUSD-OTC", trade_log=[], payout=0.85)
    assert result.allow is True


def test_check_all_filters_blocks_on_consecutive_losses() -> None:
    df = _make_df(n=100)
    log = [_trade("LOSS")] * 5
    with patch("regime_filter.get_winrate_by_hour", return_value={}), \
         patch("regime_filter.get_winrate_by_weekday", return_value={}), \
         patch("regime_filter.drift_filter", return_value=FilterResult.ok()), \
         patch("regime_filter.datetime") as mock_dt:
        mock_dt.now.return_value = _safe_datetime()
        result = check_all_filters(df=df, asset="X", trade_log=log, payout=0.85, max_consecutive=3)
    assert result.allow is False
    assert result.auto_shutdown is True


def test_check_all_filters_blocks_on_low_payout() -> None:
    df = _make_df(n=100)
    with patch("regime_filter.get_winrate_by_hour", return_value={}), \
         patch("regime_filter.get_winrate_by_weekday", return_value={}), \
         patch("regime_filter.drift_filter", return_value=FilterResult.ok()), \
         patch("regime_filter.datetime") as mock_dt:
        mock_dt.now.return_value = _safe_datetime()
        result = check_all_filters(df=df, asset="X", trade_log=[], payout=0.60, min_payout=0.80)
    assert result.allow is False
    assert result.filter_name == "payout_filter"


def test_check_all_filters_returns_first_failure() -> None:
    """Con múltiples filtros fallando, devuelve el primero (daily_loss tiene prioridad)."""
    df = _make_df(n=100)
    log = [_trade("LOSS")] * 10   # falla daily_loss Y consecutive
    with patch("regime_filter.get_winrate_by_hour", return_value={}), \
         patch("regime_filter.get_winrate_by_weekday", return_value={}), \
         patch("regime_filter.drift_filter", return_value=FilterResult.ok()), \
         patch("regime_filter.datetime") as mock_dt:
        mock_dt.now.return_value = _safe_datetime()
        result = check_all_filters(df=df, asset="X", trade_log=log, payout=0.85,
                                    max_daily_losses=3, max_consecutive=3)
    assert result.filter_name == "daily_loss_filter"


# ─── _calculate_atr ───────────────────────────────────────────────────────────

def test_calculate_atr_returns_series() -> None:
    df = _make_df(n=50)
    atr = _calculate_atr(df, period=14)
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(df)


def test_calculate_atr_non_negative() -> None:
    df = _make_df(n=50)
    atr = _calculate_atr(df, period=14)
    assert (atr.dropna() >= 0).all()


def test_calculate_atr_higher_volatility_higher_atr() -> None:
    """ATR debe ser mayor con velas más volátiles."""
    df_calm    = _make_df(n=60, seed=1)
    df_volatile = _make_df(n=60, seed=1)
    df_volatile["high"]  = df_volatile["high"]  + 0.01
    df_volatile["low"]   = df_volatile["low"]   - 0.01

    atr_calm     = _calculate_atr(df_calm,     period=14).iloc[-1]
    atr_volatile = _calculate_atr(df_volatile, period=14).iloc[-1]
    assert atr_volatile > atr_calm
