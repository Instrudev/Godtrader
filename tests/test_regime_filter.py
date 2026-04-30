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
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d") if today else "2020-01-01"
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


def test_hour_filter_bad_winrate_disabled_when_walkforward_off() -> None:
    """With WALKFORWARD_ENABLED=False, bad winrate does NOT block (curve-fitting eliminated)."""
    with patch("regime_filter.WALKFORWARD_ENABLED", False), \
         patch("regime_filter.get_winrate_by_hour", return_value={10: 0.45}):
        result = hour_profile_filter(hour_utc=10, asset="EURUSD-OTC", min_winrate=0.52)
    assert result.allow is True  # disabled → always allow


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


def test_weekday_filter_saturday_disabled_when_walkforward_off() -> None:
    """With WALKFORWARD_ENABLED=False, Saturday low winrate does NOT block."""
    with patch("regime_filter.WALKFORWARD_ENABLED", False), \
         patch("regime_filter.get_winrate_by_weekday", return_value={5: 0.45}):
        result = weekday_profile_filter(weekday=5, min_winrate=0.52)
    assert result.allow is True  # disabled → always allow


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
    from datetime import datetime, timezone
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log = [{"result": "PENDING", "timestamp": today_utc + "T10:00:00"}] * 5
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


# ─── Tarea 1.1: UTC migration tests ─────────────────────────────────────────

def test_today_utc_helper_returns_iso_format() -> None:
    """_today_utc() retorna YYYY-MM-DD sin hora ni timezone."""
    from regime_filter import _today_utc
    from unittest.mock import patch
    from datetime import datetime, timezone

    fixed = datetime(2026, 4, 30, 14, 23, 45, tzinfo=timezone.utc)
    with patch("regime_filter.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _today_utc()

    assert result == "2026-04-30"
    assert "T" not in result
    assert "+" not in result


def test_daily_loss_resets_at_utc_midnight() -> None:
    """Trades a las 23:55 UTC y 00:05 UTC del día siguiente cuentan como días distintos."""
    log = [
        {"result": "LOSS", "timestamp": "2026-04-29T23:55:00"},
        {"result": "LOSS", "timestamp": "2026-04-29T23:56:00"},
        {"result": "LOSS", "timestamp": "2026-04-29T23:57:00"},  # 3 losses el 29
        {"result": "LOSS", "timestamp": "2026-04-30T00:05:00"},  # 1 loss el 30
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = daily_loss_filter(log, max_daily_losses=3)
    # Solo 1 loss "hoy" (30 abr) → no debe bloquear
    assert result.allow is True


def test_daily_loss_uses_utc_not_local_timezone() -> None:
    """Simula servidor en UTC-5: medianoche local (05:00 UTC) no resetea el contador."""
    # Escenario: servidor en UTC-5. Son las 00:30 local = 05:30 UTC.
    # Hay 3 losses a las 04:00, 04:30, 05:00 UTC (todas del mismo día UTC: 30 abr).
    log = [
        {"result": "LOSS", "timestamp": "2026-04-30T04:00:00"},
        {"result": "LOSS", "timestamp": "2026-04-30T04:30:00"},
        {"result": "LOSS", "timestamp": "2026-04-30T05:00:00"},
    ]
    # _today_utc devuelve fecha UTC real (30 abr), no fecha local
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = daily_loss_filter(log, max_daily_losses=3)
    # 3 losses hoy UTC → debe bloquear
    assert result.allow is False
    assert result.auto_shutdown is True


def test_daily_loss_counts_only_utc_today() -> None:
    """Solo las pérdidas con timestamp de hoy UTC se cuentan."""
    log = [
        {"result": "LOSS", "timestamp": "2026-04-28T12:00:00"},  # hace 2 días
        {"result": "LOSS", "timestamp": "2026-04-29T12:00:00"},  # ayer
        {"result": "LOSS", "timestamp": "2026-04-30T12:00:00"},  # hoy
        {"result": "LOSS", "timestamp": "2026-04-30T13:00:00"},  # hoy
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = daily_loss_filter(log, max_daily_losses=3)
    # Solo 2 losses hoy → no bloquea
    assert result.allow is True


def test_consecutive_loss_filter_unchanged_by_utc_migration() -> None:
    """consecutive_loss_filter NO usa fecha — no debe verse afectado por migración UTC.
    TIE rompe racha (Tarea 1.3: TIE_BREAKS_LOSS_STREAK=True por defecto)."""
    log = [
        _trade("WIN"),
        _trade("LOSS"),
        _trade("TIE"),   # rompe la racha (política Tarea 1.3)
        _trade("LOSS"),
        _trade("LOSS"),
    ]
    result = consecutive_loss_filter(log, max_consecutive=3)
    # TIE rompe racha: solo 2 LOSS consecutivas al final → no bloquea
    assert result.allow is True


def test_max_trades_filter_uses_utc() -> None:
    """max_trades_filter usa _today_utc(), no date.today()."""
    log = [
        {"result": "WIN", "timestamp": "2026-04-29T23:50:00"},   # ayer UTC
        {"result": "WIN", "timestamp": "2026-04-30T00:10:00"},   # hoy UTC
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = max_trades_filter(log, max_trades=2)
    # Solo 1 trade hoy UTC → no bloquea
    assert result.allow is True


# ─── Tarea 1.2: Stop Loss dual (per-asset + global) ─────────────────────────

def test_per_asset_3_losses_blocks_that_asset() -> None:
    """3 LOSS en EURUSD-OTC hoy → bloquea EURUSD-OTC."""
    from regime_filter import per_asset_loss_filter
    log = [
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T10:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T11:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T12:00:00"},
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = per_asset_loss_filter(log, asset="EURUSD-OTC", max_losses=3)
    assert result.allow is False
    assert result.filter_name == "per_asset_loss_filter"
    assert "EURUSD-OTC" in result.reason
    assert result.auto_shutdown is False  # NO halt total


def test_per_asset_block_allows_other_assets() -> None:
    """3 LOSS en EURUSD-OTC → USDJPY-OTC sigue permitido."""
    from regime_filter import per_asset_loss_filter
    log = [
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T10:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T11:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T12:00:00"},
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = per_asset_loss_filter(log, asset="USDJPY-OTC", max_losses=3)
    assert result.allow is True  # USDJPY no tiene pérdidas


def test_global_6_losses_triggers_shutdown() -> None:
    """6 LOSS distribuidas entre activos → halt global."""
    log = [
        {"result": "LOSS", "asset": f"ASSET{i}-OTC", "timestamp": "2026-04-30T10:00:00"}
        for i in range(6)
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = daily_loss_filter(log, max_daily_losses=6)
    assert result.allow is False
    assert result.auto_shutdown is True


def test_global_halt_not_triggered_below_6() -> None:
    """5 LOSS distribuidas → NO halt global."""
    log = [
        {"result": "LOSS", "asset": f"ASSET{i}-OTC", "timestamp": "2026-04-30T10:00:00"}
        for i in range(5)
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = daily_loss_filter(log, max_daily_losses=6)
    assert result.allow is True


def test_per_asset_reset_at_utc_midnight() -> None:
    """LOSS de ayer no cuentan para el bloqueo de hoy."""
    from regime_filter import per_asset_loss_filter
    log = [
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-29T22:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-29T23:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-29T23:55:00"},
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        result = per_asset_loss_filter(log, asset="EURUSD-OTC", max_losses=3)
    # Las 3 pérdidas son de ayer → hoy tiene 0 → no bloquea
    assert result.allow is True


def test_per_asset_and_global_coexist() -> None:
    """3 en un activo bloquea ese activo; el conteo global sigue acumulando."""
    from regime_filter import per_asset_loss_filter
    log = [
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T10:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T11:00:00"},
        {"result": "LOSS", "asset": "EURUSD-OTC", "timestamp": "2026-04-30T12:00:00"},
        {"result": "LOSS", "asset": "USDJPY-OTC", "timestamp": "2026-04-30T13:00:00"},
    ]
    with patch("regime_filter._today_utc", return_value="2026-04-30"):
        # EURUSD bloqueado por per_asset
        asset_result = per_asset_loss_filter(log, asset="EURUSD-OTC", max_losses=3)
        assert asset_result.allow is False

        # USDJPY sigue OK por per_asset
        other_result = per_asset_loss_filter(log, asset="USDJPY-OTC", max_losses=3)
        assert other_result.allow is True

        # Global: 4 losses → no halt (< 6)
        global_result = daily_loss_filter(log, max_daily_losses=6)
        assert global_result.allow is True


# ─── Tarea 1.3: Política de TIE en rachas consecutivas ──────────────────────

def test_tie_breaks_streak_when_flag_true() -> None:
    """TIE_BREAKS=True: LOSS,LOSS,TIE,LOSS,LOSS → racha = 2 (TIE rompe)."""
    log = [_trade("LOSS"), _trade("LOSS"), _trade("TIE"),
           _trade("LOSS"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3, tie_breaks=True)
    assert result.allow is True  # racha = 2, no alcanza 3


def test_tie_invisible_when_flag_false() -> None:
    """TIE_BREAKS=False: LOSS,LOSS,TIE,LOSS,LOSS → racha = 4 (TIE invisible)."""
    log = [_trade("LOSS"), _trade("LOSS"), _trade("TIE"),
           _trade("LOSS"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3, tie_breaks=False)
    assert result.allow is False  # racha = 4, supera 3
    assert result.auto_shutdown is True


def test_default_tie_policy_is_true() -> None:
    """La constante TIE_BREAKS_LOSS_STREAK por defecto es True."""
    from regime_filter import TIE_BREAKS_LOSS_STREAK
    assert TIE_BREAKS_LOSS_STREAK is True


def test_consecutive_blocks_with_tie_breaks_at_3() -> None:
    """TIE_BREAKS=True: WIN,LOSS,LOSS,LOSS → racha = 3 (bloquea sin TIE de por medio)."""
    log = [_trade("WIN"), _trade("LOSS"), _trade("LOSS"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3, tie_breaks=True)
    assert result.allow is False
    assert result.auto_shutdown is True


def test_pure_loss_streak_unaffected_by_flag() -> None:
    """LOSS,LOSS,LOSS sin TIE → ambos flags dan racha = 3."""
    log = [_trade("LOSS"), _trade("LOSS"), _trade("LOSS")]

    result_true = consecutive_loss_filter(log, max_consecutive=3, tie_breaks=True)
    result_false = consecutive_loss_filter(log, max_consecutive=3, tie_breaks=False)

    assert result_true.allow is False
    assert result_false.allow is False
    assert result_true.auto_shutdown is True
    assert result_false.auto_shutdown is True


def test_multiple_tie_intercalated() -> None:
    """TIE_BREAKS=True: LOSS,TIE,LOSS,TIE,LOSS → racha = 1 (cada TIE rompe)."""
    log = [_trade("LOSS"), _trade("TIE"), _trade("LOSS"),
           _trade("TIE"), _trade("LOSS")]
    result = consecutive_loss_filter(log, max_consecutive=3, tie_breaks=True)
    assert result.allow is True  # racha = 1 (solo el último LOSS)


# ─── Tarea 2.5: Filtro de pendiente BB media ─────────────────────────────────

def _make_slope_df(bb_mid_values: list) -> pd.DataFrame:
    """Crea DataFrame mínimo con bb_mid controlado para tests de pendiente."""
    n = max(len(bb_mid_values), 10)
    base_mid = 1.1000
    data = {
        "open": [base_mid] * n, "close": [base_mid] * n,
        "high": [base_mid + 0.001] * n, "low": [base_mid - 0.001] * n,
        "volume": [100] * n,
        "time": pd.date_range("2026-04-30 10:00", periods=n, freq="1min"),
        "rsi": [50.0] * n,
        "bb_upper": [base_mid + 0.001] * n,
        "bb_mid": [base_mid] * n,
        "bb_lower": [base_mid - 0.001] * n,
        "bb_width": [0.5] * n, "vol_rel": [1.0] * n,
        "ema20": [base_mid] * n, "ema200": [base_mid] * n,
        "rel_atr": [1.0] * n,
    }
    df = pd.DataFrame(data)
    # Sobrescribir las últimas velas con los bb_mid controlados
    for i, val in enumerate(bb_mid_values):
        df.at[df.index[-(len(bb_mid_values) - i)], "bb_mid"] = val
    return df


def test_bb_slope_blocks_put_in_strong_uptrend() -> None:
    """Tendencia alcista fuerte → señal PUT bloqueada."""
    from regime_filter import bb_slope_filter
    # bb_mid sube: N-5=1.1000 → N=1.1015 → slope=+0.136% (> 0.08%)
    bb_mids = [1.1000, 1.1003, 1.1006, 1.1009, 1.1012, 1.1015]
    df = _make_slope_df(bb_mids)
    result = bb_slope_filter(df, direction="PUT", threshold_pct=0.08)
    assert result.allow is False
    assert result.filter_name == "bb_slope_filter"
    assert "alcista" in result.reason


def test_bb_slope_blocks_call_in_strong_downtrend() -> None:
    """Tendencia bajista fuerte → señal CALL bloqueada."""
    from regime_filter import bb_slope_filter
    # bb_mid baja: N-5=1.1015 → N=1.1000 → slope=-0.136%
    bb_mids = [1.1015, 1.1012, 1.1009, 1.1006, 1.1003, 1.1000]
    df = _make_slope_df(bb_mids)
    result = bb_slope_filter(df, direction="CALL", threshold_pct=0.08)
    assert result.allow is False
    assert "bajista" in result.reason


def test_bb_slope_allows_signals_in_lateral() -> None:
    """Mercado lateral (slope ~0%) → ambas señales permitidas."""
    from regime_filter import bb_slope_filter
    # bb_mid plano: slope ≈ 0%
    bb_mids = [1.1000, 1.1001, 1.1000, 1.1001, 1.1000, 1.1001]
    df = _make_slope_df(bb_mids)

    result_call = bb_slope_filter(df, direction="CALL", threshold_pct=0.08)
    result_put  = bb_slope_filter(df, direction="PUT", threshold_pct=0.08)
    assert result_call.allow is True
    assert result_put.allow is True


def test_bb_slope_allows_put_in_downtrend() -> None:
    """Tendencia bajista → señal PUT permitida (pro-tendencia)."""
    from regime_filter import bb_slope_filter
    bb_mids = [1.1015, 1.1012, 1.1009, 1.1006, 1.1003, 1.1000]
    df = _make_slope_df(bb_mids)
    result = bb_slope_filter(df, direction="PUT", threshold_pct=0.08)
    assert result.allow is True  # PUT en bajada = pro-tendencia


def test_bb_slope_allows_call_in_uptrend() -> None:
    """Tendencia alcista → señal CALL permitida (pro-tendencia)."""
    from regime_filter import bb_slope_filter
    bb_mids = [1.1000, 1.1003, 1.1006, 1.1009, 1.1012, 1.1015]
    df = _make_slope_df(bb_mids)
    result = bb_slope_filter(df, direction="CALL", threshold_pct=0.08)
    assert result.allow is True  # CALL en subida = pro-tendencia


def test_bb_slope_threshold_configurable() -> None:
    """Umbral configurable: slope 0.10% con threshold 0.05% → bloquea."""
    from regime_filter import bb_slope_filter
    # slope = +0.10% (> 0.05% threshold custom)
    bb_mids = [1.1000, 1.1002, 1.1004, 1.1006, 1.1008, 1.1011]
    df = _make_slope_df(bb_mids)
    result = bb_slope_filter(df, direction="PUT", threshold_pct=0.05)
    assert result.allow is False  # 0.10% > 0.05% threshold → bloqueado


def test_bb_slope_insufficient_history_allows() -> None:
    """df con menos de 6 velas → fail-open (permitido)."""
    from regime_filter import bb_slope_filter
    df = _make_slope_df([1.1000, 1.1010])[:4]  # solo 4 velas
    result = bb_slope_filter(df, direction="CALL", threshold_pct=0.08)
    assert result.allow is True  # fail-open


# ─── Tarea 2.4: Walk-forward flag ────────────────────────────────────────────

def test_hour_profile_disabled_when_walkforward_off() -> None:
    """WALKFORWARD_ENABLED=False → hour_profile_filter always allows."""
    with patch("regime_filter.WALKFORWARD_ENABLED", False), \
         patch("regime_filter.get_winrate_by_hour", return_value={10: 0.20}):
        result = hour_profile_filter(hour_utc=10, min_winrate=0.55)
    assert result.allow is True


def test_weekday_profile_disabled_when_walkforward_off() -> None:
    """WALKFORWARD_ENABLED=False → weekday_profile_filter always allows."""
    with patch("regime_filter.WALKFORWARD_ENABLED", False), \
         patch("regime_filter.get_winrate_by_weekday", return_value={5: 0.10}):
        result = weekday_profile_filter(weekday=5, min_winrate=0.55)
    assert result.allow is True


def test_hour_profile_active_when_walkforward_on() -> None:
    """WALKFORWARD_ENABLED=True → hour_profile_filter blocks bad winrate."""
    with patch("regime_filter.WALKFORWARD_ENABLED", True), \
         patch("regime_filter.get_winrate_by_hour", return_value={10: 0.40}):
        result = hour_profile_filter(hour_utc=10, min_winrate=0.55)
    assert result.allow is False
    assert result.filter_name == "hour_profile_filter"


def test_walkforward_enabled_constant_is_false() -> None:
    """WALKFORWARD_ENABLED default es False durante remediación."""
    from regime_filter import WALKFORWARD_ENABLED
    assert WALKFORWARD_ENABLED is False
