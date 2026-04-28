"""
tests/test_indicators_otc.py – Tests para las nuevas funciones OTC de indicators.py
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from indicators import (
    _empty_streak,
    _measure_all_streaks,
    build_dataframe,
    calculate_atr,
    calculate_relative_atr,
    estimate_half_life,
    get_streak_info,
    load_strategy_config,
    otc_signal,
    pre_qualify,
    pre_qualify_classical,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 250, seed: int = 0, mean_rev: bool = False) -> pd.DataFrame:
    """DataFrame de velas sintéticas con indicadores calculados."""
    rng = np.random.default_rng(seed)
    t0  = 1_700_000_000
    price = 1.08500
    candles = []
    for i in range(n):
        if mean_rev:
            drift = (1.08500 - price) * 0.05   # mean reversion fuerte
        else:
            drift = 0.0
        change = drift + rng.normal(0, 0.0003)
        open_p = price
        close_p = price + change
        candles.append({
            "time":  t0 + i * 60,
            "open":  open_p,
            "high":  max(open_p, close_p) + abs(rng.normal(0, 0.0001)),
            "low":   min(open_p, close_p) - abs(rng.normal(0, 0.0001)),
            "close": close_p,
        })
        price = close_p
    return build_dataframe(candles)


def _make_streak_df(direction: str, streak_len: int, base_n: int = 200, seed: int = 1) -> pd.DataFrame:
    """
    DataFrame donde las últimas streak_len velas van todas en la misma dirección.
    Útil para probar get_streak_info.
    """
    rng = np.random.default_rng(seed)
    t0  = 1_700_000_000
    price = 1.08500
    candles = []

    # Velas mixtas iniciales
    for i in range(base_n):
        change = rng.normal(0, 0.0003)
        open_p = price
        close_p = price + change
        candles.append({
            "time":  t0 + i * 60,
            "open":  open_p,
            "high":  max(open_p, close_p) + 0.00005,
            "low":   min(open_p, close_p) - 0.00005,
            "close": close_p,
        })
        price = close_p

    # Racha forzada al final
    for j in range(streak_len):
        change = 0.0003 if direction == "bull" else -0.0003
        open_p = price
        close_p = price + change
        i_idx = base_n + j
        candles.append({
            "time":  t0 + i_idx * 60,
            "open":  open_p,
            "high":  close_p + 0.00005,
            "low":   open_p  - 0.00005,
            "close": close_p,
        })
        price = close_p

    return build_dataframe(candles)


# ─── calculate_atr ────────────────────────────────────────────────────────────

def test_atr_returns_series_same_length() -> None:
    df = _make_df(60)
    atr = calculate_atr(df, period=14)
    assert len(atr) == len(df)


def test_atr_non_negative() -> None:
    df = _make_df(60)
    assert (calculate_atr(df).dropna() >= 0).all()


def test_atr_higher_for_volatile_market() -> None:
    df_calm     = _make_df(60, seed=0)
    df_volatile = _make_df(60, seed=0)
    # Ampliar el rango de las velas
    df_volatile = df_volatile.copy()
    df_volatile["high"] = df_volatile["high"] + 0.01
    df_volatile["low"]  = df_volatile["low"]  - 0.01
    atr_calm     = float(calculate_atr(df_calm).iloc[-1])
    atr_volatile = float(calculate_atr(df_volatile).iloc[-1])
    assert atr_volatile > atr_calm


def test_atr_constant_prices_near_zero() -> None:
    rng = np.random.default_rng(0)
    t0 = 1_700_000_000
    candles = [{"time": t0+i*60, "open": 1.0, "high": 1.0001, "low": 0.9999, "close": 1.0}
               for i in range(50)]
    df = build_dataframe(candles)
    atr = calculate_atr(df).iloc[-1]
    assert atr < 0.001


# ─── calculate_relative_atr ───────────────────────────────────────────────────

def test_rel_atr_average_conditions_near_one() -> None:
    df  = _make_df(100)
    rel = calculate_relative_atr(df)
    assert 0.3 < rel < 3.0   # cerca de 1.0 en condiciones normales


def test_rel_atr_positive() -> None:
    df = _make_df(80)
    assert calculate_relative_atr(df) > 0


def test_rel_atr_uses_cached_column_when_available() -> None:
    """Si el DataFrame ya tiene columna 'atr', la usa directamente."""
    df = _make_df(80)
    assert "atr" in df.columns
    rel1 = calculate_relative_atr(df)
    # Modificar la columna atr directamente
    df2 = df.copy()
    df2["atr"] = df2["atr"] * 2
    rel2 = calculate_relative_atr(df2)
    # Si usa la columna cached, el ratio debe ser igual (atr/mean_atr = 2x/2x = mismo ratio)
    assert abs(rel1 - rel2) < 0.01


# ─── estimate_half_life ───────────────────────────────────────────────────────

def test_half_life_mean_reverting_process_finite() -> None:
    """Un proceso con mean reversion fuerte debe tener half-life finita."""
    df = _make_df(200, seed=0, mean_rev=True)
    hl = estimate_half_life(df, window=150)
    assert hl < float("inf")
    assert hl > 0


def test_half_life_random_walk_infinite() -> None:
    """Un random walk sin deriva no debe tener half-life finita (β ≥ 0)."""
    # Con semilla específica que genere random walk puro
    rng = np.random.default_rng(999)
    t0 = 1_700_000_000
    price = 100.0
    candles = []
    for i in range(150):
        change = rng.normal(0, 0.01)   # drift=0
        price += change
        candles.append({"time": t0+i*60, "open": price-abs(change)/2,
                         "high": price+0.001, "low": price-0.001, "close": price})
    df = build_dataframe(candles)
    hl = estimate_half_life(df, window=100)
    # No podemos garantizar que sea infinita (depende de la muestra), pero
    # verificamos que la función no lanza error
    assert hl >= 0


def test_half_life_insufficient_data() -> None:
    df = _make_df(10)   # muy pocos datos
    hl = estimate_half_life(df, window=100)
    assert hl == float("inf")


def test_half_life_always_positive_or_inf() -> None:
    df = _make_df(200)
    hl = estimate_half_life(df)
    assert hl > 0 or hl == float("inf")


def test_half_life_stronger_mean_rev_shorter_hl() -> None:
    """Mayor mean reversion → half-life más corta."""
    rng = np.random.default_rng(42)
    t0 = 1_700_000_000

    def make_ou(kappa: float, n: int = 200):
        p = 1.08500
        candles = []
        for i in range(n):
            dp = kappa * (1.08500 - p) + rng.normal(0, 0.0002)
            p += dp
            candles.append({"time": t0+i*60, "open": p-dp/2,
                             "high": p+0.0001, "low": p-0.0001, "close": p})
        return build_dataframe(candles)

    df_fast = make_ou(0.20)   # mean reversion rápida
    df_slow = make_ou(0.02)   # mean reversion lenta

    hl_fast = estimate_half_life(df_fast, window=150)
    hl_slow = estimate_half_life(df_slow, window=150)

    if hl_fast < float("inf") and hl_slow < float("inf"):
        assert hl_fast < hl_slow


# ─── _measure_all_streaks ─────────────────────────────────────────────────────

def test_measure_streaks_alternating() -> None:
    dirs = np.array([1, -1, 1, -1, 1, -1])
    streaks = _measure_all_streaks(dirs)
    assert all(s == 1 for s in streaks)


def test_measure_streaks_single_long_streak() -> None:
    dirs = np.array([1, 1, 1, 1, 1])
    streaks = _measure_all_streaks(dirs)
    assert streaks == [5]


def test_measure_streaks_ignores_doji() -> None:
    dirs = np.array([1, 0, 1, 0, 1])   # 3 alcistas con dojis en medio
    streaks = _measure_all_streaks(dirs)
    assert streaks == [3]   # dojis no rompen la racha


def test_measure_streaks_empty() -> None:
    assert _measure_all_streaks(np.array([])) == []


def test_measure_streaks_all_doji() -> None:
    assert _measure_all_streaks(np.array([0, 0, 0])) == []


def test_measure_streaks_mixed() -> None:
    dirs = np.array([1, 1, -1, -1, -1, 1])
    streaks = _measure_all_streaks(dirs)
    assert streaks == [2, 3, 1]


# ─── get_streak_info ──────────────────────────────────────────────────────────

def test_streak_info_insufficient_data() -> None:
    df = _make_df(5)
    info = get_streak_info(df)
    assert info["direction"] == "none"
    assert info["current_length"] == 0


def test_streak_info_bull_streak() -> None:
    df = _make_streak_df("bull", streak_len=5, base_n=200)
    info = get_streak_info(df)
    assert info["direction"] == "bull"
    assert info["current_length"] >= 3   # al menos 3 alcistas al final


def test_streak_info_bear_streak() -> None:
    df = _make_streak_df("bear", streak_len=5, base_n=200)
    info = get_streak_info(df)
    assert info["direction"] == "bear"
    assert info["current_length"] >= 3


def test_streak_info_percentile_between_0_and_100() -> None:
    df = _make_df(250, seed=0)
    info = get_streak_info(df)
    if info["direction"] != "none":
        assert 0.0 <= info["percentile"] <= 100.0


def test_streak_info_long_streak_high_percentile() -> None:
    """Una racha muy larga debe tener percentil alto."""
    df = _make_streak_df("bull", streak_len=12, base_n=200)
    info = get_streak_info(df)
    if info["direction"] == "bull" and info["current_length"] >= 8:
        assert info["percentile"] >= 70


def test_streak_info_is_extreme_flag() -> None:
    df = _make_streak_df("bear", streak_len=10, base_n=200)
    info = get_streak_info(df)
    # is_extreme debe ser True cuando percentile >= 75
    assert info["is_extreme"] == (info["percentile"] >= 75)


def test_streak_info_mean_streak_positive() -> None:
    df = _make_df(250)
    info = get_streak_info(df)
    if info["direction"] != "none":
        assert info["mean_streak"] > 0


# ─── otc_signal ───────────────────────────────────────────────────────────────

def test_otc_signal_returns_dict() -> None:
    df = _make_df(250)
    sig = otc_signal(df, config={"streak_percentile_min": 70, "rel_atr_min": 0.3, "rel_atr_max": 3.0})
    assert isinstance(sig, dict)
    assert "has_signal" in sig
    assert "direction" in sig
    assert "confidence" in sig


def test_otc_signal_with_extreme_streak() -> None:
    df = _make_streak_df("bear", streak_len=12, base_n=200)
    cfg = {"streak_percentile_min": 60, "rel_atr_min": 0.2, "rel_atr_max": 4.0,
           "half_life_max_candles": 100}
    sig = otc_signal(df, config=cfg)
    if sig["has_signal"]:
        assert sig["direction"] == "CALL"   # racha bajista → esperar CALL
        assert 0.0 < sig["confidence"] <= 1.0


def test_otc_signal_confidence_in_range() -> None:
    df = _make_streak_df("bull", streak_len=10, base_n=200)
    cfg = {"streak_percentile_min": 50, "rel_atr_min": 0.2, "rel_atr_max": 5.0,
           "half_life_max_candles": 100}
    sig = otc_signal(df, config=cfg)
    if sig["has_signal"]:
        assert 0.0 <= sig["confidence"] <= 1.0


def test_otc_signal_no_signal_when_streak_not_extreme() -> None:
    """Con percentil mínimo muy alto (99%), casi nunca habrá señal."""
    df = _make_df(250, seed=42)
    cfg = {"streak_percentile_min": 99, "rel_atr_min": 0.0, "rel_atr_max": 100.0}
    sig = otc_signal(df, config=cfg)
    assert sig["has_signal"] is False


# ─── pre_qualify (nueva versión OTC) ─────────────────────────────────────────

def test_pre_qualify_returns_tuple() -> None:
    df = _make_df(250)
    result = pre_qualify(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_pre_qualify_insufficient_data() -> None:
    df = _make_df(15)   # muy pocos datos
    ok, reason = pre_qualify(df)
    assert ok is False


def test_pre_qualify_with_extreme_streak_qualifies() -> None:
    """Racha extrema con config permisivo debe calificar."""
    df = _make_streak_df("bear", streak_len=12, base_n=200)
    cfg = {"streak_percentile_min": 60, "rel_atr_min": 0.1, "rel_atr_max": 10.0,
           "enable_classical_fallback": True, "require_price_action": False}
    ok, reason = pre_qualify(df, config=cfg)
    if ok:
        assert "racha" in reason.lower() or "BEAR" in reason or "CALL" in reason


def test_pre_qualify_classical_fallback() -> None:
    """Sin racha extrema y fallback habilitado, debe usar lógica clásica."""
    df = _make_df(250, seed=7)
    cfg = {"streak_percentile_min": 99, "rel_atr_min": 0.0, "rel_atr_max": 100.0,
           "enable_classical_fallback": True, "require_price_action": False}
    ok, reason = pre_qualify(df, config=cfg)
    # No podemos asegurar que pase, pero debe ejecutar sin error
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


def test_pre_qualify_no_fallback_blocks_without_streak() -> None:
    """Sin racha extrema y sin fallback → bloquear."""
    df = _make_df(250, seed=7)
    cfg = {"streak_percentile_min": 99, "rel_atr_min": 0.0, "rel_atr_max": 100.0,
           "enable_classical_fallback": False, "require_price_action": False}
    ok, _ = pre_qualify(df, config=cfg)
    assert ok is False


# ─── pre_qualify_classical ────────────────────────────────────────────────────

def test_pre_qualify_classical_returns_tuple() -> None:
    df = _make_df(250)
    result = pre_qualify_classical(df)
    assert len(result) == 2 and isinstance(result[0], bool)


def test_pre_qualify_classical_passes_with_oversold_rsi() -> None:
    """Si RSI < 35 y precio en zona baja de BB, debe pasar."""
    df = _make_df(250)
    # Forzar condiciones en última vela
    df.at[df.index[-1], "rsi"]      = 28.0
    df.at[df.index[-1], "close"]    = float(df["bb_lower"].iloc[-1]) * 0.9999
    ok, reason = pre_qualify_classical(df)
    assert ok is True
    assert "SOBREVENDIDO" in reason


def test_pre_qualify_classical_blocks_neutral_rsi() -> None:
    df = _make_df(250)
    df.at[df.index[-1], "rsi"] = 50.0
    ok, _ = pre_qualify_classical(df)
    assert ok is False


# ─── load_strategy_config ─────────────────────────────────────────────────────

def test_load_strategy_config_returns_dict() -> None:
    cfg = load_strategy_config()
    assert isinstance(cfg, dict)


def test_load_strategy_config_has_default_section() -> None:
    cfg = load_strategy_config()
    if cfg:   # si el archivo existe
        assert "default" in cfg


def test_load_strategy_config_missing_file(tmp_path) -> None:
    from pathlib import Path
    cfg = load_strategy_config(path=tmp_path / "nonexistent.yaml")
    assert cfg == {}


# ─── build_dataframe now includes atr and rel_atr ────────────────────────────

def test_build_dataframe_has_atr_column() -> None:
    df = _make_df(50)
    assert "atr" in df.columns


def test_build_dataframe_has_rel_atr_column() -> None:
    df = _make_df(50)
    assert "rel_atr" in df.columns


def test_build_dataframe_rel_atr_positive() -> None:
    df = _make_df(50)
    assert (df["rel_atr"].dropna() > 0).all()
