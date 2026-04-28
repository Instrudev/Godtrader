"""
tests/test_drift_detector.py – Tests unitarios para generator_drift_detector.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from generator_drift_detector import (
    AUTOCORR_DELTA,
    CURRENT_WINDOW_SIZE,
    F_RATIO_THRESHOLD,
    KS_THRESHOLD,
    MIN_WINDOW_SIZE,
    REFERENCE_WINDOW_SIZE,
    Z_THRESHOLD,
    DriftDetector,
    DriftResult,
    DriftState,
    _autocorr_delta,
    _candles_to_returns,
    _f_ratio_variances,
    _ks_statistic,
    _z_test_means,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_candles(n: int, mu: float = 0.0, sigma: float = 0.0003, seed: int = 42) -> list:
    """Genera velas sintéticas con retornos normales N(mu, sigma)."""
    rng = np.random.default_rng(seed)
    price = 1.08500
    candles = []
    t0 = 1_700_000_000
    for i in range(n):
        ret = rng.normal(mu, sigma)
        price = price * np.exp(ret)
        candles.append({
            "time":  t0 + i * 60,
            "open":  price,
            "high":  price * 1.0002,
            "low":   price * 0.9998,
            "close": price,
        })
    return candles


@pytest.fixture
def detector(tmp_path: Path) -> DriftDetector:
    """DriftDetector con estado temporal limpio."""
    return DriftDetector(path=tmp_path / "drift_state.json")


@pytest.fixture
def candles_stable() -> list:
    return _make_candles(400, mu=0.0, sigma=0.0003, seed=1)


@pytest.fixture
def candles_drifted() -> list:
    """Feed con propiedades estadísticas radicalmente distintas."""
    # Media diferente (drift en media) + varianza 5x mayor
    return _make_candles(400, mu=0.0010, sigma=0.0015, seed=2)


# ─── _candles_to_returns ──────────────────────────────────────────────────────

def test_candles_to_returns_length() -> None:
    candles = _make_candles(50)
    returns = _candles_to_returns(candles)
    assert len(returns) == 49   # n-1 retornos


def test_candles_to_returns_no_inf_nan() -> None:
    candles = _make_candles(100)
    returns = _candles_to_returns(candles)
    assert np.all(np.isfinite(returns))


def test_candles_to_returns_empty() -> None:
    returns = _candles_to_returns([])
    assert len(returns) == 0


def test_candles_to_returns_single_candle() -> None:
    candles = _make_candles(1)
    returns = _candles_to_returns(candles)
    assert len(returns) == 0


def test_candles_to_returns_filters_zero_close() -> None:
    candles = _make_candles(10)
    candles[5]["close"] = 0  # precio inválido
    returns = _candles_to_returns(candles)
    assert np.all(np.isfinite(returns))


# ─── _ks_statistic ────────────────────────────────────────────────────────────

def test_ks_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    ks = _ks_statistic(a, b)
    # Dos muestras de la misma distribución → KS bajo
    assert ks < 0.15


def test_ks_different_distributions_high() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0,   0.001, 200)
    b = rng.normal(0.1, 0.001, 200)   # media muy diferente
    ks = _ks_statistic(a, b)
    assert ks > 0.5


def test_ks_range_zero_to_one() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 100)
    b = rng.normal(1, 1, 100)
    ks = _ks_statistic(a, b)
    assert 0.0 <= ks <= 1.0


def test_ks_symmetric() -> None:
    """KS(a, b) ≈ KS(b, a)."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 150)
    b = rng.normal(0.05, 1.2, 150)
    assert abs(_ks_statistic(a, b) - _ks_statistic(b, a)) < 1e-10


# ─── _z_test_means ────────────────────────────────────────────────────────────

def test_z_test_same_mean_near_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 1, 200)
    z = _z_test_means(a, b)
    assert abs(z) < 3.0   # con n=200, dos muestras iguales raramente superan 3


def test_z_test_different_means_high() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0,  0.001, 200)
    b = rng.normal(0.05, 0.001, 200)
    z = _z_test_means(a, b)
    assert abs(z) > Z_THRESHOLD


def test_z_test_insufficient_data() -> None:
    z = _z_test_means(np.array([1.0]), np.array([2.0]))
    assert z == 0.0


def test_z_test_sign_reflects_direction() -> None:
    rng = np.random.default_rng(5)
    ref = rng.normal(0,    0.001, 200)
    cur = rng.normal(0.01, 0.001, 200)  # media mayor
    z = _z_test_means(ref, cur)
    assert z > 0


# ─── _f_ratio_variances ───────────────────────────────────────────────────────

def test_f_ratio_same_variance_near_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0, 1, 300)
    f = _f_ratio_variances(a, b)
    assert 0.5 < f < 2.0


def test_f_ratio_higher_variance_greater_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 3, 200)   # 3x más varianza
    f = _f_ratio_variances(a, b)
    assert f > F_RATIO_THRESHOLD


def test_f_ratio_always_positive() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 0.5, 100)
    b = rng.normal(0, 2.0, 100)
    assert _f_ratio_variances(a, b) > 0


def test_f_ratio_zero_ref_variance_returns_one() -> None:
    """Si la referencia tiene varianza cero, devolver 1 (fail-safe)."""
    a = np.ones(50)   # varianza = 0
    b = np.random.default_rng(0).normal(0, 1, 50)
    assert _f_ratio_variances(a, b) == 1.0


# ─── _autocorr_delta ─────────────────────────────────────────────────────────

def test_autocorr_delta_same_process_near_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0, 1, 300)
    delta = _autocorr_delta(a, b)
    assert abs(delta) < 0.3


def test_autocorr_delta_different_autocorrelation() -> None:
    rng = np.random.default_rng(0)
    # Serie con fuerte autocorrelación vs ruido blanco
    ar = np.zeros(300)
    ar[0] = rng.normal()
    for i in range(1, 300):
        ar[i] = 0.8 * ar[i-1] + rng.normal(0, 0.5)
    white = rng.normal(0, 1, 200)
    delta = _autocorr_delta(white, ar[:200])
    assert abs(delta) > AUTOCORR_DELTA


def test_autocorr_delta_insufficient_data() -> None:
    delta = _autocorr_delta(np.array([1.0]), np.array([2.0]))
    assert delta == 0.0


# ─── DriftDetector – persistencia ────────────────────────────────────────────

def test_detector_load_no_file(tmp_path: Path) -> None:
    d = DriftDetector.load(path=tmp_path / "nonexistent.json")
    assert d.state.ref_returns == {}
    assert d.state.drift_alerts == []


def test_detector_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    d1 = DriftDetector(path=path)
    d1.state.ref_returns["TEST"] = [0.001, -0.002, 0.003]
    d1.save()

    d2 = DriftDetector.load(path=path)
    assert d2.state.ref_returns["TEST"] == pytest.approx([0.001, -0.002, 0.003])


def test_detector_load_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("no es json válido {{{")
    d = DriftDetector.load(path=path)
    assert d.state.ref_returns == {}   # falla con gracia


# ─── DriftDetector – analyze ─────────────────────────────────────────────────

def test_analyze_insufficient_data(detector: DriftDetector) -> None:
    candles = _make_candles(10)   # menos de MIN_WINDOW_SIZE
    result = detector.analyze(candles, "EURUSD-OTC")
    assert result is None


def test_analyze_first_run_establishes_reference(
    detector: DriftDetector, candles_stable: list
) -> None:
    result = detector.analyze(candles_stable, "EURUSD-OTC")
    assert result is not None
    assert result.drift_detected is False
    assert "EURUSD-OTC" in detector.state.ref_returns


def test_analyze_stable_data_no_drift(
    detector: DriftDetector, candles_stable: list
) -> None:
    # Primera llamada: establece referencia
    detector.analyze(candles_stable, "EURUSD-OTC")
    # Segunda llamada: mismos datos → no drift
    result = detector.analyze(candles_stable, "EURUSD-OTC")
    assert result is not None
    assert result.drift_detected is False


def test_analyze_drifted_data_detects_drift(
    detector: DriftDetector, candles_stable: list, candles_drifted: list
) -> None:
    # Establecer referencia con datos estables
    detector.analyze(candles_stable, "EURUSD-OTC")
    # Analizar datos con drift
    result = detector.analyze(candles_drifted, "EURUSD-OTC")
    assert result is not None
    assert result.drift_detected is True
    assert len(result.triggers) >= 2


def test_analyze_drift_saved_to_alerts(
    detector: DriftDetector, candles_stable: list, candles_drifted: list
) -> None:
    detector.analyze(candles_stable, "EURUSD-OTC")
    detector.analyze(candles_drifted, "EURUSD-OTC")
    assert len(detector.state.drift_alerts) >= 1
    assert detector.state.drift_alerts[-1]["asset"] == "EURUSD-OTC"


# ─── DriftDetector – has_drift ───────────────────────────────────────────────

def test_has_drift_no_alerts_false(detector: DriftDetector) -> None:
    assert detector.has_drift("EURUSD-OTC") is False


def test_has_drift_recent_alert_true(detector: DriftDetector) -> None:
    detector.state.drift_alerts.append({
        "asset":     "EURUSD-OTC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggers":  ["KS=0.25"],
        "ks_stat":   0.25,
    })
    assert detector.has_drift("EURUSD-OTC", days=7) is True


def test_has_drift_old_alert_false(detector: DriftDetector) -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    detector.state.drift_alerts.append({
        "asset":     "EURUSD-OTC",
        "timestamp": old_ts,
        "triggers":  ["KS=0.25"],
        "ks_stat":   0.25,
    })
    assert detector.has_drift("EURUSD-OTC", days=7) is False


def test_has_drift_different_asset_false(detector: DriftDetector) -> None:
    detector.state.drift_alerts.append({
        "asset":     "GBPJPY-OTC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "triggers":  ["KS=0.25"],
        "ks_stat":   0.25,
    })
    assert detector.has_drift("EURUSD-OTC") is False


# ─── DriftDetector – clear_drift ─────────────────────────────────────────────

def test_clear_drift_removes_alerts(detector: DriftDetector) -> None:
    detector.state.drift_alerts = [
        {"asset": "EURUSD-OTC", "timestamp": "2026-04-20T10:00:00", "triggers": [], "ks_stat": 0.2},
        {"asset": "GBPJPY-OTC", "timestamp": "2026-04-20T10:00:00", "triggers": [], "ks_stat": 0.1},
    ]
    detector.clear_drift("EURUSD-OTC")
    assert not any(a["asset"] == "EURUSD-OTC" for a in detector.state.drift_alerts)
    assert any(a["asset"] == "GBPJPY-OTC" for a in detector.state.drift_alerts)


# ─── DriftResult ─────────────────────────────────────────────────────────────

def test_drift_result_str_no_drift() -> None:
    r = DriftResult(asset="EURUSD-OTC", timestamp="2026-04-20T10:00:00+00:00",
                    drift_detected=False)
    text = str(r)
    assert "Sin drift" in text


def test_drift_result_str_with_drift() -> None:
    r = DriftResult(asset="EURUSD-OTC", timestamp="2026-04-20T10:00:00+00:00",
                    drift_detected=True, triggers=["KS=0.25", "Z=3.1"])
    text = str(r)
    assert "DRIFT DETECTADO" in text
