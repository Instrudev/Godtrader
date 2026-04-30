"""
tests/test_scanner_filters.py – Tests de integración de filtros en asset_scanner (Tarea 1.0).

Valida que check_all_filters está correctamente integrado en _try_execute()
y que los 13 filtros de régimen se evalúan antes de ML.

6 tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_candidate(asset: str = "EURUSD-OTC", direction: str = "CALL") -> dict:
    """Crea un candidato mínimo para _try_execute."""
    n = 50
    np.random.seed(42)
    prices = 1.1000 + np.random.randn(n).cumsum() * 0.0001
    times = pd.date_range("2026-04-30 10:00", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 0.0002,
        "low": prices - 0.0002,
        "close": prices,
        "volume": np.random.randint(50, 200, n),
        "time": times,
        "rsi": np.full(n, 50.0),
        "bb_upper": prices + 0.001,
        "bb_mid": prices,
        "bb_lower": prices - 0.001,
        "bb_width": np.full(n, 0.5),
        "vol_rel": np.full(n, 1.0),
        "ema20": prices,
        "ema200": prices,
        "rel_atr": np.full(n, 1.0),
    })
    return {"asset": asset, "df": df, "direction": direction, "reason": "test", "score": 0.5}


@pytest.fixture
def scanner():
    """AssetScanner con dependencias mockeadas."""
    from asset_scanner import AssetScanner
    s = AssetScanner()
    s.running = True
    s.mode = "paper"
    s.amount = 1.0
    s.expiry_min = 2
    s.asset_type = "binary"
    s.trade_log = []
    s.on_notification = None
    s.loop = None
    return s


# ─── Test 1: check_all_filters se invoca ─────────────────────────────────────

def test_scanner_calls_check_all_filters(scanner):
    """check_all_filters se invoca en _try_execute."""
    candidate = _make_candidate()

    with patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        # Filtro bloquea — no importa qué devuelve ML
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block("test_filter", "test reason")

        result = scanner._try_execute(candidate)

    assert result is False
    mock_caf.assert_called_once()
    # Verificar que se pasaron los parámetros correctos
    call_kwargs = mock_caf.call_args
    assert call_kwargs.kwargs["asset"] == "EURUSD-OTC"
    assert call_kwargs.kwargs["direction"] == "CALL"
    assert call_kwargs.kwargs["payout"] == 0.85


# ─── Test 2: filtro bloquea → ML no se invoca ───────────────────────────────

def test_scanner_filter_blocks_before_ml(scanner):
    """Si check_all_filters bloquea, ML no se invoca."""
    candidate = _make_candidate()

    with patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq, \
         patch("asset_scanner.ml_classifier") as mock_ml:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block("blocked_hours_filter", "hora bloqueada")

        result = scanner._try_execute(candidate)

    assert result is False
    # ML predict_proba NUNCA se llamó
    mock_ml.predict_proba.assert_not_called()


# ─── Test 3: filtro pasa → ML se invoca ─────────────────────────────────────

def test_scanner_filter_passes_allows_ml(scanner):
    """Si check_all_filters pasa, ML se invoca normalmente."""
    candidate = _make_candidate()

    with patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq, \
         patch("asset_scanner.ml_classifier") as mock_ml, \
         patch("asset_scanner.extract_features", return_value={"rsi": 50.0}), \
         patch("asset_scanner.DB_PATH", "test.db"):
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.ok()

        # ML cargado y suficientes trades
        mock_ml.is_loaded.return_value = True
        mock_ml.predict_proba.return_value = {"call_proba": 0.30, "put_proba": 0.30}

        with patch("database.fetch_trades", return_value=[{"result": "WIN"}] * 200):
            result = scanner._try_execute(candidate)

    # ML sí se invocó (aunque rechazó por proba baja)
    mock_ml.predict_proba.assert_called_once()
    assert result is False  # rechazado por ML, no por filtro


# ─── Test 4: auto_shutdown detiene el scanner ────────────────────────────────

def test_scanner_auto_shutdown_stops_scanner(scanner):
    """auto_shutdown=True setea self.running = False."""
    candidate = _make_candidate()
    assert scanner.running is True

    with patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block(
            "daily_loss_filter", "6 pérdidas hoy", shutdown=True
        )

        result = scanner._try_execute(candidate)

    assert result is False
    assert scanner.running is False  # scanner detenido


# ─── Test 5: loss_pattern_filter no se llama dos veces ───────────────────────

def test_scanner_no_double_loss_pattern(scanner):
    """loss_pattern_filter solo se evalúa dentro de check_all_filters, no por separado."""
    candidate = _make_candidate()

    with patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block("loss_pattern_filter", "patrón detectado")

        with patch("regime_filter.loss_pattern_filter") as mock_lpf:
            result = scanner._try_execute(candidate)

    # loss_pattern_filter NO se llama directamente desde _try_execute
    # (solo se evalúa dentro de check_all_filters)
    mock_lpf.assert_not_called()


# ─── Test 6: payout se calcula antes de filtros ─────────────────────────────

def test_scanner_payout_calculated_before_filters(scanner):
    """Payout se calcula antes de invocar check_all_filters y se pasa correctamente."""
    candidate = _make_candidate()
    expected_payout = 0.82

    with patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = expected_payout
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block("payout_filter", "payout bajo")

        scanner._try_execute(candidate)

    # Verificar que check_all_filters recibió el payout correcto
    call_kwargs = mock_caf.call_args.kwargs
    assert call_kwargs["payout"] == expected_payout
    # Verificar que get_payout se llamó ANTES (implícito: si payout llega a check_all_filters,
    # se calculó antes)
    mock_iq.get_payout.assert_called_once()
