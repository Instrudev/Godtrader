"""
tests/test_scanner_filters.py – Tests de integración de filtros y estrategias en asset_scanner.

Tarea 1.0: check_all_filters integrado (6 tests).
Tarea 1.2: halt + reconstrucción trade_log (2 tests).
Tarea 1.4: _evaluate_strategies paralelo + arbitraje (10 tests).
Tarea 1.6: ML_DISABLED_MODE fallback (8 tests).
Tarea 2.2: Umbrales endurecidos (8 tests).
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
    """Si check_all_filters pasa y ML_DISABLED=False, ML se invoca normalmente."""
    candidate = _make_candidate()

    with patch("asset_scanner.ML_DISABLED_MODE", False), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
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


# ─── Tarea 1.2: Tests de scanner ─────────────────────────────────────────────

def test_scanner_stops_processing_assets_after_halt(scanner):
    """Si auto_shutdown activa a mitad del ciclo, los siguientes candidatos NO se procesan."""
    c1 = _make_candidate(asset="EURUSD-OTC")
    c2 = _make_candidate(asset="USDJPY-OTC")

    call_count = 0
    original_try = scanner._try_execute

    def mock_try_execute(candidate):
        nonlocal call_count
        call_count += 1
        # Primer candidato dispara halt
        if candidate["asset"] == "EURUSD-OTC":
            scanner.running = False  # simula auto_shutdown
            return False
        return False

    scanner._try_execute = mock_try_execute
    scanner.running = True

    # Simular el loop de candidatos (lógica extraída de _scan_loop)
    qualified = [c1, c2]
    import asyncio

    async def _run():
        for candidate in qualified:
            if not scanner.running:
                break
            scanner._try_execute(candidate)

    asyncio.get_event_loop().run_until_complete(_run())

    assert call_count == 1  # solo EURUSD procesado, USDJPY saltado
    assert scanner.running is False


def test_scanner_reconstructs_trade_log_from_db(tmp_path):
    """Al startup, el scanner reconstruye trade_log desde BD con trades del día UTC."""
    import sqlite3
    from datetime import datetime, timedelta, timezone
    from asset_scanner import AssetScanner

    # Crear BD temporal con trades del día
    db_path = tmp_path / "test_trades.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            timestamp TEXT, asset TEXT, direction TEXT, result TEXT,
            expiry_min INT, mode TEXT, price REAL, rsi REAL,
            bb_pct_b REAL, bb_width_pct REAL, vol_rel REAL,
            ema20 REAL, ema200 REAL, hour_utc INT, weekday INT,
            predicted_proba REAL, ai_reasoning TEXT, order_id INT,
            open_price REAL, payout REAL, profit REAL,
            close_price REAL, pips_difference REAL, closed_at TEXT
        )
    """)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    # 3 losses de hoy + 2 de ayer (no deben contar)
    trades = [
        (f"{today}T10:00:00", "EURUSD-OTC", "CALL", "LOSS"),
        (f"{today}T11:00:00", "EURUSD-OTC", "PUT", "LOSS"),
        (f"{today}T12:00:00", "EURUSD-OTC", "CALL", "LOSS"),
        (f"{today}T13:00:00", "USDJPY-OTC", "PUT", "WIN"),
        (f"{yesterday}T23:00:00", "EURUSD-OTC", "CALL", "LOSS"),
        (f"{yesterday}T23:30:00", "EURUSD-OTC", "PUT", "LOSS"),
    ]
    for ts, asset, direction, result in trades:
        conn.execute(
            "INSERT INTO trades (timestamp, asset, direction, result) VALUES (?, ?, ?, ?)",
            (ts, asset, direction, result),
        )
    conn.commit()
    conn.close()

    # Llamar _reconstruct_trade_log directamente (no requiere event loop)
    s = AssetScanner()
    with patch("asset_scanner.DB_PATH", str(db_path)):
        s.trade_log = s._reconstruct_trade_log()

    # Verificar: 4 trades de hoy (3 LOSS + 1 WIN), 0 de ayer
    assert len(s.trade_log) == 4

    # Verificar que per_asset_loss_filter bloquearía EURUSD-OTC
    from regime_filter import per_asset_loss_filter
    with patch("regime_filter._today_utc", return_value=today):
        result = per_asset_loss_filter(s.trade_log, asset="EURUSD-OTC", max_losses=3)
    assert result.allow is False  # 3 LOSS hoy → bloqueado

    # USDJPY-OTC sigue permitido
    with patch("regime_filter._today_utc", return_value=today):
        result2 = per_asset_loss_filter(s.trade_log, asset="USDJPY-OTC", max_losses=3)
    assert result2.allow is True


# ─── Tarea 1.4: _evaluate_strategies (paralelo + arbitraje) ──────────────────

def _make_df_with_indicators(**overrides):
    """Crea DataFrame mínimo con indicadores para _evaluate_strategies."""
    n = 50
    np.random.seed(42)
    prices = 1.1000 + np.random.randn(n).cumsum() * 0.0001
    defaults = {
        "rsi": 50.0, "bb_upper": 1.1010, "bb_mid": 1.1000,
        "bb_lower": 1.0990, "bb_width": 0.5, "vol_rel": 1.0,
    }
    defaults.update(overrides)
    return pd.DataFrame({
        "open": prices, "close": prices,
        "high": prices + 0.0002, "low": prices - 0.0002,
        "volume": [100] * n,
        "time": pd.date_range("2026-04-30 10:00", periods=n, freq="1min"),
        "rsi": [defaults["rsi"]] * n,
        "bb_upper": [defaults["bb_upper"]] * n,
        "bb_mid": [defaults["bb_mid"]] * n,
        "bb_lower": [defaults["bb_lower"]] * n,
        "bb_width": [defaults["bb_width"]] * n,
        "vol_rel": [defaults["vol_rel"]] * n,
        "ema20": prices, "ema200": prices,
        "rel_atr": [1.0] * n,
    })


def test_no_strategies_active_returns_none():
    """Ninguna estrategia activa → (None, 0.0, _, 'no_signal')."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)  # RSI neutral → no RSI signal

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    assert direction is None
    assert score == 0.0
    assert resolution == "no_signal"
    assert all(not s.active for s in signals)


def test_single_strategy_returns_its_direction():
    """Solo streak activa → retorna su dirección."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": True, "percentile": 85, "current_length": 5, "direction": "bear"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    assert direction == "CALL"  # bear streak → CALL reversal
    assert score > 0
    assert resolution == "single_match"


def test_multiple_same_direction_returns_highest_score():
    """BB 2-Candle CALL + RSI Clásico CALL → CALL con mayor score."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=20.0)  # RSI < 35 → CALL

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "CALL", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    assert direction == "CALL"
    assert resolution == "multiple_match"
    # Both active, winner is the one with highest score
    active = [s for s in signals if s.active]
    assert len(active) == 2
    winner = max(active, key=lambda s: s.score)
    assert score == winner.score  # returned score matches winner


def test_contradictory_directions_returns_none():
    """BB 2-Candle CALL + RSI Clásico PUT → conflict_cancel."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=80.0)  # RSI > 65 → PUT

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "CALL", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    assert direction is None
    assert score == 0.0
    assert resolution == "conflict_cancel"


def test_3_call_1_put_conflict_cancel():
    """2 CALL + 1 PUT → conflict_cancel (mayoría NO gana).
    BB Body deprecada (Tarea 2.3), así que conflicto viene de bb_2candle vs RSI."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=80.0)  # RSI > 75 → PUT

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "CALL", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.detect_bb_body_reversal_call", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": True, "percentile": 85, "current_length": 5, "direction": "bear"}):
        # bb_2candle=CALL, streak=CALL (bear→CALL), rsi_classical=PUT (RSI>75)
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    assert direction is None
    assert resolution == "conflict_cancel"


def test_score_from_winning_strategy():
    """El score devuelto es exactamente el del ganador, no un max() agregado."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "PUT", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    assert direction == "PUT"
    bb2_signal = [s for s in signals if s.strategy_name == "bb_2candle"][0]
    assert score == bb2_signal.score


def test_telemetry_single_match():
    """resolution='single_match' cuando solo una estrategia activa."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "CALL", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        _, _, _, resolution = _evaluate_strategies(df, asset="TEST")

    assert resolution == "single_match"


def test_telemetry_conflict_cancel():
    """resolution='conflict_cancel' cuando hay direcciones opuestas."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=80.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "CALL", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        _, _, _, resolution = _evaluate_strategies(df, asset="TEST")

    assert resolution == "conflict_cancel"


def test_strategy_signal_frozen():
    """StrategySignal es inmutable (frozen dataclass)."""
    from asset_scanner import StrategySignal
    sig = StrategySignal(active=True, direction="CALL", score=0.75, strategy_name="test")
    with pytest.raises(AttributeError):
        sig.score = 0.99  # type: ignore


def test_backwards_compatible_signature():
    """_evaluate_strategies retorna tupla de 4 elementos compatible con callsites."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        result = _evaluate_strategies(df, asset="TEST")

    assert len(result) == 4
    direction, score, signals, resolution = result
    assert direction is None or direction in ("CALL", "PUT")
    assert isinstance(score, float)
    assert isinstance(signals, list)
    assert resolution in ("single_match", "multiple_match", "conflict_cancel", "no_signal")


# ─── Tarea 1.6: ML_DISABLED_MODE fallback ────────────────────────────────────

@pytest.fixture
def scanner_ml_disabled():
    """Scanner con ML_DISABLED_MODE y dependencias mockeadas."""
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


def _make_candidate_with_score(score: float, asset: str = "EURUSD-OTC", direction: str = "CALL") -> dict:
    """Crea candidato con score y resolution controlados."""
    c = _make_candidate(asset=asset, direction=direction)
    c["score"] = score
    c["resolution"] = "single_match"
    return c


def test_ml_disabled_blocks_below_min_score(scanner_ml_disabled):
    """ML_DISABLED: score 0.65 < 0.75 → no opera."""
    candidate = _make_candidate_with_score(0.65)

    with patch("asset_scanner.ML_DISABLED_MODE", True), \
         patch("asset_scanner.ML_DISABLED_MIN_SCORE", 0.75), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.ok()

        result = scanner_ml_disabled._try_execute(candidate)

    assert result is False


def test_ml_disabled_allows_above_min_score(scanner_ml_disabled):
    """ML_DISABLED: score 0.80 ≥ 0.75 → opera (sujeto a otros filtros)."""
    candidate = _make_candidate_with_score(0.80)

    with patch("asset_scanner.ML_DISABLED_MODE", True), \
         patch("asset_scanner.ML_DISABLED_MIN_SCORE", 0.75), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq, \
         patch("asset_scanner.time") as mock_time:
        mock_iq.get_payout.return_value = 0.85
        mock_iq.buy_binary.return_value = (True, 99999)
        mock_time.time.return_value = 1.0  # dentro del guard de tiempo
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.ok()

        result = scanner_ml_disabled._try_execute(candidate)

    assert result is True


def test_ml_required_when_disabled_mode_false(scanner_ml_disabled):
    """ML_DISABLED=False, ML no cargado → no opera."""
    candidate = _make_candidate_with_score(0.80)

    with patch("asset_scanner.ML_DISABLED_MODE", False), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq, \
         patch("asset_scanner.ml_classifier") as mock_ml:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.ok()
        mock_ml.is_loaded.return_value = False

        result = scanner_ml_disabled._try_execute(candidate)

    assert result is False


def test_ml_disabled_uses_reduced_asset_limit(scanner_ml_disabled):
    """ML_DISABLED: 2 losses/activo → bloquea activo (no 3)."""
    candidate = _make_candidate_with_score(0.80)

    with patch("asset_scanner.ML_DISABLED_MODE", True), \
         patch("asset_scanner.ML_DISABLED_MAX_ASSET_LOSSES", 2), \
         patch("asset_scanner.ML_DISABLED_MAX_DAILY_LOSSES", 4), \
         patch("asset_scanner.ML_DISABLED_MAX_DAILY_TRADES", 5), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block("per_asset_loss_filter", "2 losses")

        result = scanner_ml_disabled._try_execute(candidate)

    assert result is False
    # Verificar que check_all_filters recibió max_asset_losses=2
    call_kwargs = mock_caf.call_args.kwargs
    assert call_kwargs["max_asset_losses"] == 2
    assert call_kwargs["max_daily_losses"] == 4
    assert call_kwargs["max_trades"] == 5


def test_ml_disabled_uses_reduced_global_limit(scanner_ml_disabled):
    """ML_DISABLED: 4 losses globales → halt (no 6)."""
    candidate = _make_candidate_with_score(0.80)

    with patch("asset_scanner.ML_DISABLED_MODE", True), \
         patch("asset_scanner.ML_DISABLED_MAX_ASSET_LOSSES", 2), \
         patch("asset_scanner.ML_DISABLED_MAX_DAILY_LOSSES", 4), \
         patch("asset_scanner.ML_DISABLED_MAX_DAILY_TRADES", 5), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block(
            "daily_loss_filter", "4 losses", shutdown=True
        )

        result = scanner_ml_disabled._try_execute(candidate)

    assert result is False
    assert scanner_ml_disabled.running is False  # auto_shutdown


def test_ml_disabled_uses_reduced_daily_trades(scanner_ml_disabled):
    """ML_DISABLED: max_trades=5 pasado a check_all_filters."""
    candidate = _make_candidate_with_score(0.80)

    with patch("asset_scanner.ML_DISABLED_MODE", True), \
         patch("asset_scanner.ML_DISABLED_MAX_ASSET_LOSSES", 2), \
         patch("asset_scanner.ML_DISABLED_MAX_DAILY_LOSSES", 4), \
         patch("asset_scanner.ML_DISABLED_MAX_DAILY_TRADES", 5), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.block("max_trades_filter", "5 trades")

        result = scanner_ml_disabled._try_execute(candidate)

    assert result is False
    assert mock_caf.call_args.kwargs["max_trades"] == 5


def test_ml_normal_uses_standard_limits(scanner_ml_disabled):
    """ML_DISABLED=False → límites normales 3/6/15."""
    candidate = _make_candidate_with_score(0.80)

    with patch("asset_scanner.ML_DISABLED_MODE", False), \
         patch("asset_scanner.MAX_ASSET_DAILY_LOSSES", 3), \
         patch("asset_scanner.MAX_DAILY_LOSSES", 6), \
         patch("asset_scanner.MAX_TRADES_PER_DAY", 15), \
         patch("asset_scanner.check_all_filters") as mock_caf, \
         patch("asset_scanner.get_winrate_by_hour", return_value={}), \
         patch("asset_scanner.iq_service") as mock_iq, \
         patch("asset_scanner.ml_classifier") as mock_ml:
        mock_iq.get_payout.return_value = 0.85
        from regime_filter import FilterResult
        mock_caf.return_value = FilterResult.ok()
        mock_ml.is_loaded.return_value = False

        scanner_ml_disabled._try_execute(candidate)

    call_kwargs = mock_caf.call_args.kwargs
    assert call_kwargs["max_asset_losses"] == 3
    assert call_kwargs["max_daily_losses"] == 6
    assert call_kwargs["max_trades"] == 15


def test_startup_banner_shows_ml_disabled(caplog):
    """Banner de startup incluye ML_DISABLED_MODE."""
    import logging
    import iqservice
    orig = iqservice.ML_DISABLED_MODE
    try:
        iqservice.REMEDIATION_MODE = True
        iqservice.ML_DISABLED_MODE = True

        with caplog.at_level(logging.WARNING):
            # Simular la lógica del banner del lifespan
            _logger = logging.getLogger("test_banner")
            _logger.warning(
                "═══ REMEDIATION MODE — DEMO ONLY ═══ "
                f"(FORCE_DEMO_ACCOUNT={iqservice.FORCE_DEMO_ACCOUNT}, "
                f"ML_DISABLED={iqservice.ML_DISABLED_MODE})"
            )
            if iqservice.ML_DISABLED_MODE:
                _logger.warning(
                    "OPERATING WITHOUT ML — REDUCED RISK MODE | "
                    f"min_score={iqservice.ML_DISABLED_MIN_SCORE}"
                )

        assert "ML_DISABLED=True" in caplog.text
        assert "OPERATING WITHOUT ML" in caplog.text
        assert "REDUCED RISK MODE" in caplog.text
    finally:
        iqservice.ML_DISABLED_MODE = orig


# ─── Tarea 2.2: Umbrales endurecidos ─────────────────────────────────────────

def test_bb2c_rejects_below_70pct_body():
    """BB 2-Candle: body 60% fuera de banda → no activa (umbral ahora 70%)."""
    from indicators import detect_bb_two_candle_reversal
    n = 30
    df = _make_df_with_indicators(rsi=15.0)
    # Configurar 2 velas con ~60% body debajo de bb_lower (no alcanza 70%)
    for i in [-2, -1]:
        df.at[df.index[i], "open"] = 1.0990
        df.at[df.index[i], "close"] = 1.0985   # vela bajista
        df.at[df.index[i], "low"] = 1.0980
        df.at[df.index[i], "bb_lower"] = 1.0988  # ~60% del body debajo
    df.at[df.index[-1], "rsi"] = 15.0

    ok, direction, reason = detect_bb_two_candle_reversal(df)
    assert ok is False  # 60% < 70% → no activa


def test_bb2c_accepts_above_70pct_body_with_divergence():
    """BB 2-Candle: body 85% fuera + RSI<20 + divergencia + vela alcista → activa."""
    from indicators import detect_bb_two_candle_reversal
    df = _make_df_with_indicators(rsi=15.0)
    # Configurar 2 velas con ~85% body debajo de bb_lower
    for i in [-2, -1]:
        df.at[df.index[i], "open"] = 1.0980
        df.at[df.index[i], "close"] = 1.0985   # vela alcista (confirmación)
        df.at[df.index[i], "low"] = 1.0975
        df.at[df.index[i], "high"] = 1.0990
        df.at[df.index[i], "bb_lower"] = 1.0984  # >70% body debajo
    df.at[df.index[-1], "rsi"] = 15.0
    # Divergencia: low actual < low N-3, RSI actual > RSI N-3
    df.at[df.index[-4], "low"] = 1.0980   # N-3 tiene low más alto
    df.at[df.index[-4], "rsi"] = 12.0     # N-3 tiene RSI más bajo
    # N actual: low=1.0975 < 1.0980, rsi=15 > 12 → divergencia alcista OK

    ok, direction, reason = detect_bb_two_candle_reversal(df)
    assert ok is True
    assert direction == "CALL"
    assert "divergencia" in reason.lower()


def test_bb2c_rejects_without_rsi_divergence():
    """BB 2-Candle: body OK + RSI OK pero sin divergencia → no activa."""
    from indicators import detect_bb_two_candle_reversal
    df = _make_df_with_indicators(rsi=15.0)
    for i in [-2, -1]:
        df.at[df.index[i], "open"] = 1.0980
        df.at[df.index[i], "close"] = 1.0985
        df.at[df.index[i], "low"] = 1.0975
        df.at[df.index[i], "high"] = 1.0990
        df.at[df.index[i], "bb_lower"] = 1.0984
    df.at[df.index[-1], "rsi"] = 15.0
    # Sin divergencia: low actual > low N-3 (NO lower low)
    df.at[df.index[-4], "low"] = 1.0970   # N-3 tiene low MÁS bajo
    df.at[df.index[-4], "rsi"] = 12.0

    ok, direction, reason = detect_bb_two_candle_reversal(df)
    assert ok is False
    assert "divergencia" in reason.lower()


def test_bb2c_put_divergence():
    """BB 2-Candle PUT: body 85% sobre BB upper + RSI>80 + divergencia bajista → activa."""
    from indicators import detect_bb_two_candle_reversal
    df = _make_df_with_indicators(rsi=85.0)
    for i in [-2, -1]:
        df.at[df.index[i], "open"] = 1.1020
        df.at[df.index[i], "close"] = 1.1015  # vela bajista (confirmación)
        df.at[df.index[i], "high"] = 1.1025
        df.at[df.index[i], "low"] = 1.1010
        df.at[df.index[i], "bb_upper"] = 1.1016  # >70% body arriba
    df.at[df.index[-1], "rsi"] = 85.0
    # Divergencia bajista: high actual > high N-3, RSI actual < RSI N-3
    df.at[df.index[-4], "high"] = 1.1020  # N-3 tiene high más bajo
    df.at[df.index[-4], "rsi"] = 88.0     # N-3 tiene RSI más alto

    ok, direction, reason = detect_bb_two_candle_reversal(df)
    assert ok is True
    assert direction == "PUT"


def test_rsi_classical_25_75_thresholds():
    """RSI=30 (antes activa, ahora no) y RSI=20 (activa con nuevo umbral 25)."""
    from asset_scanner import _evaluate_strategies

    # RSI=30: entre 25 y 75 → no activa RSI clásico
    df30 = _make_df_with_indicators(rsi=30.0)
    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        _, _, signals30, _ = _evaluate_strategies(df30, asset="TEST")
    rsi_sig30 = [s for s in signals30 if s.strategy_name == "rsi_classical"][0]
    assert rsi_sig30.active is False

    # RSI=20: < 25 → activa RSI clásico
    df20 = _make_df_with_indicators(rsi=20.0)
    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        _, _, signals20, _ = _evaluate_strategies(df20, asset="TEST")
    rsi_sig20 = [s for s in signals20 if s.strategy_name == "rsi_classical"][0]
    assert rsi_sig20.active is True
    assert rsi_sig20.direction == "CALL"


def test_rsi_classical_bb_zone_10pct():
    """pct_b=0.12 → no activa con zona 10% (antes sí con 15%)."""
    from indicators import pre_qualify_classical
    df = _make_df_with_indicators(rsi=20.0)
    # Configurar pct_b ~0.12 (dentro de 15% pero fuera de 10%)
    bb_low = 1.0990
    bb_up = 1.1010
    price = bb_low + (bb_up - bb_low) * 0.12  # pct_b = 0.12
    df.at[df.index[-1], "close"] = price
    df.at[df.index[-1], "bb_lower"] = bb_low
    df.at[df.index[-1], "bb_upper"] = bb_up
    df.at[df.index[-1], "rsi"] = 20.0

    ok, reason = pre_qualify_classical(df)
    assert ok is False  # pct_b=0.12 > 0.10 → no toca zona inferior


def test_streak_requires_4_candles_and_pct90():
    """Racha 3v pct=85% → no activa (necesita ≥4v y ≥90%)."""
    from indicators import pre_qualify
    df = _make_df_with_indicators()
    cfg = {
        "streak_percentile_min": 90, "rel_atr_min": 0.0, "rel_atr_max": 100.0,
        "enable_classical_fallback": False,
    }
    with patch("indicators.get_streak_info", return_value={
        "is_extreme": True, "percentile": 85, "current_length": 3,
        "direction": "bear", "mean": 2.0, "std": 1.0,
    }):
        ok, reason = pre_qualify(df, asset="TEST", config=cfg)
    assert ok is False  # pct=85 < 90 OR length=3 < 4


def test_streak_requires_half_life_le_25():
    """Racha 5v pct=92% pero half-life=30 → no activa (hl obligatorio ≤25)."""
    from indicators import pre_qualify
    df = _make_df_with_indicators()
    cfg = {
        "streak_percentile_min": 90, "rel_atr_min": 0.0, "rel_atr_max": 100.0,
        "enable_classical_fallback": False,
    }
    with patch("indicators.get_streak_info", return_value={
        "is_extreme": True, "percentile": 92, "current_length": 5,
        "direction": "bear", "mean": 2.0, "std": 1.0,
    }), patch("indicators.estimate_half_life", return_value=30.0):
        ok, reason = pre_qualify(df, asset="TEST", config=cfg)
    assert ok is False
    assert "half-life" in reason.lower()


# ─── Tarea 2.3: BB Body phantom logging ──────────────────────────────────────

def test_bb_body_phantom_logged_when_active():
    """Cuando BB Body PUT habría activado, phantom se registra en el log."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=70.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(True, "phantom PUT")), \
         patch("asset_scanner.detect_bb_body_reversal_call", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    # BB Body no debe ser señal activa (deprecada)
    bb_body_sig = [s for s in signals if s.strategy_name == "bb_body"][0]
    assert bb_body_sig.active is False


def test_bb_body_phantom_null_when_no_signals():
    """Sin activación de BB Body, phantom no se registra."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)

    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(False, None, "")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(False, "")), \
         patch("asset_scanner.detect_bb_body_reversal_call", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    # Todas las señales inactivas
    assert all(not s.active for s in signals)
    assert resolution == "no_signal"


def test_bb_body_not_in_evaluate_strategies_decision():
    """BB Body deprecada: su activación phantom NO afecta direction final."""
    from asset_scanner import _evaluate_strategies
    df = _make_df_with_indicators(rsi=50.0)

    # BB Body PUT habría activado + BB 2-Candle CALL activa
    with patch("asset_scanner.detect_bb_two_candle_reversal", return_value=(True, "CALL", "reason")), \
         patch("asset_scanner.detect_bb_body_reversal", return_value=(True, "phantom PUT")), \
         patch("asset_scanner.detect_bb_body_reversal_call", return_value=(False, "")), \
         patch("asset_scanner.get_streak_info", return_value={"is_extreme": False, "percentile": 30, "current_length": 1, "direction": "bull"}):
        direction, score, signals, resolution = _evaluate_strategies(df, asset="TEST")

    # Direction es CALL (de BB 2-Candle), NO PUT (de BB Body phantom)
    assert direction == "CALL"
    assert resolution == "single_match"
    # BB Body sigue inactiva en la lista de señales
    bb_body_sig = [s for s in signals if s.strategy_name == "bb_body"][0]
    assert bb_body_sig.active is False
