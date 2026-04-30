"""
tests/test_ml_drift_detector.py – Tests del detector de drift ML (Tarea 2.1).

Alcance mínimo viable: buffer circular, stats, KL divergence.
record_result y calibración real-time diferidos a Tarea 3.1.

6 tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_drift_detector import DriftDetector


# ─── Test 1: Buffer almacena correctamente ───────────────────────────────────

def test_drift_records_predictions():
    """record_prediction almacena en buffer con campos correctos."""
    det = DriftDetector(window=100, check_every=999)  # no trigger check
    det.record_prediction(0.65, 0.35, asset="EURUSD-OTC")
    det.record_prediction(0.70, 0.30, asset="USDJPY-OTC")

    assert len(det._buffer) == 2
    assert det._prediction_count == 2

    first = det._buffer[0]
    assert first["call_proba"] == 0.65
    assert first["put_proba"] == 0.35
    assert first["asset"] == "EURUSD-OTC"
    assert "timestamp" in first


# ─── Test 2: Stats correctas ────────────────────────────────────────────────

def test_drift_computes_distribution_stats():
    """Stats de distribución calculadas correctamente."""
    det = DriftDetector(window=100, check_every=999)

    # 20 predicciones con probas conocidas
    for i in range(20):
        p = 0.50 + i * 0.01  # 0.50, 0.51, ..., 0.69
        det.record_prediction(p, 1.0 - p)

    stats = det._compute_stats()

    assert stats["count"] == 20
    assert stats["mean"] == pytest.approx(0.595, abs=0.01)
    assert stats["min"] == pytest.approx(0.50, abs=0.01)
    assert stats["max"] == pytest.approx(0.69, abs=0.01)
    assert stats["std"] > 0
    assert "p25" in stats
    assert "p50" in stats
    assert "p75" in stats


# ─── Test 3: KL divergence — distribuciones idénticas ────────────────────────

def test_drift_kl_divergence_identical():
    """Distribuciones idénticas → KL ≈ 0."""
    uniform = np.array([0.1] * 10)
    kl = DriftDetector._kl_divergence(uniform, uniform)
    assert kl == pytest.approx(0.0, abs=1e-6)


# ─── Test 4: KL divergence — distribuciones distintas ────────────────────────

def test_drift_kl_divergence_different():
    """Distribuciones claramente distintas → KL > 0."""
    # Observed: concentrada en bin 5 (0.5-0.6)
    observed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    # Baseline: uniforme
    baseline = np.array([0.1] * 10)

    kl = DriftDetector._kl_divergence(observed, baseline)

    assert kl > 0.5  # KL alta para distribuciones muy distintas
    assert np.isfinite(kl)  # epsilon previene infinito


# ─── Test 5: Alerta cuando KL > umbral ───────────────────────────────────────

def test_drift_alerts_on_high_kl():
    """KL > threshold → alerta registrada."""
    det = DriftDetector(window=50, check_every=10, kl_threshold=0.15)

    # Inyectar baseline uniforme
    det._baseline = np.array([0.1] * 10)
    det._baseline_loaded = True

    # Añadir 20 predicciones concentradas en rango estrecho (0.48-0.52)
    # para que la distribución observada difiera de la uniforme
    for i in range(20):
        det.record_prediction(0.50, 0.50, asset="TEST")

    # Forzar check
    det._check_drift()

    # Debe haber alertas (distribución concentrada vs uniforme → KL alta)
    assert len(det._alerts) > 0
    assert det._alerts[-1]["type"] == "KL_DIVERGENCE"
    assert det._alerts[-1]["kl_value"] > 0.15


# ─── Test 6: Buffer circular ────────────────────────────────────────────────

def test_drift_buffer_circular():
    """Buffer mantiene solo últimos `window` elementos."""
    det = DriftDetector(window=10, check_every=999)

    for i in range(25):
        det.record_prediction(float(i) / 100, 1.0 - float(i) / 100)

    assert len(det._buffer) == 10  # window=10, solo últimos 10
    assert det._prediction_count == 25

    # El primer elemento debe ser la predicción #15 (0-indexed: i=15)
    assert det._buffer[0]["call_proba"] == pytest.approx(0.15)
    # El último debe ser #24
    assert det._buffer[-1]["call_proba"] == pytest.approx(0.24)
