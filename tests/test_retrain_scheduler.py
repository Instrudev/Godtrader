"""
tests/test_retrain_scheduler.py – Tests del scheduler de reentrenamiento (Fase 5).

Cubre:
- _State: carga, guardado, valores por defecto
- _version_tag: formato correcto
- _should_retrain: lógica de trades nuevos
- _cleanup_old_versions: límite de versiones
- RetrainScheduler.get_status: keys correctas
- RetrainScheduler.trigger_now: cuando no está corriendo
- Integración con fetch_training_data
- _profit_factor_from_metrics: casos límite
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrain_scheduler import (
    MAX_VERSIONS,
    RetrainScheduler,
    _State,
    _cleanup_old_versions,
    _profit_factor_from_metrics,
    _version_tag,
)


# ─── _State ──────────────────────────────────────────────────────────────────

def test_state_defaults() -> None:
    s = _State()
    assert s.last_trained_at    is None
    assert s.trades_at_last_run == 0
    assert s.current_version    is None
    assert s.versions            == []


def test_state_save_and_load(tmp_path, monkeypatch) -> None:
    import retrain_scheduler as rs
    monkeypatch.setattr(rs, "STATE_PATH", tmp_path / "retrain_state.json")
    monkeypatch.setattr(rs, "MODEL_DIR",  tmp_path)

    s = _State()
    s.last_trained_at    = "2024-01-01T00:00:00+00:00"
    s.trades_at_last_run = 42
    s.current_version    = "20240101_120000"
    s.versions           = [{"version": "20240101_120000", "metrics": {}}]
    s.save()

    s2 = _State.load()
    assert s2.last_trained_at    == "2024-01-01T00:00:00+00:00"
    assert s2.trades_at_last_run == 42
    assert s2.current_version    == "20240101_120000"
    assert len(s2.versions)       == 1


def test_state_load_missing_file(tmp_path, monkeypatch) -> None:
    import retrain_scheduler as rs
    monkeypatch.setattr(rs, "STATE_PATH", tmp_path / "nonexistent.json")
    s = _State.load()
    assert s.trades_at_last_run == 0


def test_state_load_corrupt_file(tmp_path, monkeypatch) -> None:
    import retrain_scheduler as rs
    state_path = tmp_path / "corrupt.json"
    state_path.write_text("{ bad json }")
    monkeypatch.setattr(rs, "STATE_PATH", state_path)
    s = _State.load()          # no debe lanzar excepción
    assert s.trades_at_last_run == 0


# ─── _version_tag ─────────────────────────────────────────────────────────────

def test_version_tag_format() -> None:
    tag = _version_tag()
    # Formato YYYYMMDD_HHMMSS — 15 caracteres
    assert len(tag) == 15
    assert "_" in tag
    parts = tag.split("_")
    assert len(parts) == 2
    assert parts[0].isdigit() and len(parts[0]) == 8
    assert parts[1].isdigit() and len(parts[1]) == 6


def test_version_tag_unique() -> None:
    """Dos llamadas sucesivas pueden producir el mismo tag si ocurren en el mismo segundo,
    pero deben ser strings válidos."""
    t1 = _version_tag()
    t2 = _version_tag()
    assert isinstance(t1, str)
    assert isinstance(t2, str)


# ─── _profit_factor_from_metrics ─────────────────────────────────────────────

def test_pf_from_metrics_normal() -> None:
    m = {"profit_factor_65": 1.45, "auc": 0.62}
    assert _profit_factor_from_metrics(m) == pytest.approx(1.45)


def test_pf_from_metrics_none() -> None:
    assert _profit_factor_from_metrics(None) == pytest.approx(0.0)


def test_pf_from_metrics_missing_key() -> None:
    assert _profit_factor_from_metrics({"auc": 0.6}) == pytest.approx(0.0)


def test_pf_from_metrics_zero() -> None:
    assert _profit_factor_from_metrics({"profit_factor_65": 0.0}) == pytest.approx(0.0)


# ─── _cleanup_old_versions ────────────────────────────────────────────────────

def test_cleanup_keeps_max_versions(tmp_path, monkeypatch) -> None:
    import retrain_scheduler as rs
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    monkeypatch.setattr(rs, "VERSIONS_DIR", versions_dir)

    # Crear 8 versiones simuladas
    for i in range(8):
        (versions_dir / f"2024010{i}_120000").mkdir()

    _cleanup_old_versions(keep=MAX_VERSIONS)
    remaining = list(versions_dir.iterdir())
    assert len(remaining) == MAX_VERSIONS


def test_cleanup_keeps_newest(tmp_path, monkeypatch) -> None:
    """Debe conservar las versiones más recientes (orden lexicográfico = cronológico)."""
    import retrain_scheduler as rs
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    monkeypatch.setattr(rs, "VERSIONS_DIR", versions_dir)

    names = [
        "20240101_120000",
        "20240102_120000",
        "20240103_120000",
        "20240104_120000",
        "20240105_120000",
        "20240106_120000",
    ]
    for n in names:
        (versions_dir / n).mkdir()

    _cleanup_old_versions(keep=3)
    remaining = sorted(d.name for d in versions_dir.iterdir())
    assert remaining == ["20240104_120000", "20240105_120000", "20240106_120000"]


def test_cleanup_no_versions_dir(tmp_path, monkeypatch) -> None:
    import retrain_scheduler as rs
    monkeypatch.setattr(rs, "VERSIONS_DIR", tmp_path / "nonexistent_versions")
    _cleanup_old_versions()   # no debe lanzar excepción


def test_cleanup_fewer_than_max(tmp_path, monkeypatch) -> None:
    import retrain_scheduler as rs
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    monkeypatch.setattr(rs, "VERSIONS_DIR", versions_dir)

    (versions_dir / "20240101_120000").mkdir()
    (versions_dir / "20240102_120000").mkdir()

    _cleanup_old_versions(keep=MAX_VERSIONS)
    assert len(list(versions_dir.iterdir())) == 2


# ─── RetrainScheduler._should_retrain ────────────────────────────────────────

def test_should_retrain_false_no_new_trades() -> None:
    """No reentrena si no hay trades nuevos suficientes."""
    sched = RetrainScheduler(min_new_trades=50)
    sched._state.trades_at_last_run = 100

    with patch("retrain_scheduler.fetch_training_data", return_value=[{}] * 100):
        # 100 totales - 100 en último run = 0 nuevos → no reentrena
        from retrain_scheduler import fetch_training_data
        result = sched._should_retrain()
    assert result is False


def test_should_retrain_true_enough_new_trades() -> None:
    sched = RetrainScheduler(min_new_trades=50)
    sched._state.trades_at_last_run = 50

    with patch("retrain_scheduler.fetch_training_data", return_value=[{}] * 110):
        # 110 - 50 = 60 nuevos ≥ 50 → debe reentrenar
        result = sched._should_retrain()
    assert result is True


def test_should_retrain_exactly_at_threshold() -> None:
    sched = RetrainScheduler(min_new_trades=50)
    sched._state.trades_at_last_run = 60

    with patch("retrain_scheduler.fetch_training_data", return_value=[{}] * 110):
        # 110 - 60 = 50 = threshold → True
        result = sched._should_retrain()
    assert result is True


def test_should_retrain_db_error_returns_false() -> None:
    """Si la DB falla, no debe reentrenar (fail-safe)."""
    sched = RetrainScheduler(min_new_trades=10)

    with patch("retrain_scheduler.fetch_training_data", side_effect=Exception("DB error")):
        result = sched._should_retrain()
    assert result is False


# ─── RetrainScheduler.get_status ─────────────────────────────────────────────

def test_get_status_keys() -> None:
    sched = RetrainScheduler()
    status = sched.get_status()
    required = {
        "running", "check_interval_h", "min_new_trades",
        "last_check_at", "last_result", "last_trained_at",
        "trades_at_last_run", "current_version", "versions",
    }
    assert required <= set(status.keys())


def test_get_status_not_running_initially() -> None:
    sched = RetrainScheduler()
    assert sched.get_status()["running"] is False


def test_get_status_versions_limited_to_5() -> None:
    sched = RetrainScheduler()
    sched._state.versions = [{"v": str(i)} for i in range(10)]
    status = sched.get_status()
    assert len(status["versions"]) <= 5


# ─── RetrainScheduler.trigger_now ────────────────────────────────────────────

def test_trigger_now_when_not_running() -> None:
    sched = RetrainScheduler()
    msg = sched.trigger_now()
    assert isinstance(msg, str)
    assert len(msg) > 0


# ─── _run_retrain: promoción cuando no hay champion ──────────────────────────

def test_run_retrain_promotes_when_no_champion(tmp_path, monkeypatch) -> None:
    """Si no hay champion previo y el challenger cumple criterios → se promueve."""
    import retrain_scheduler as rs

    monkeypatch.setattr(rs, "MODEL_DIR",      tmp_path)
    monkeypatch.setattr(rs, "VERSIONS_DIR",   tmp_path / "versions")
    monkeypatch.setattr(rs, "STATE_PATH",     tmp_path / "state.json")
    monkeypatch.setattr(rs, "CHAMPION_MODEL", tmp_path / "lgbm_model.pkl")
    monkeypatch.setattr(rs, "CHAMPION_CALIB", tmp_path / "platt_calibrator.pkl")

    # Mock train para que diga que el challenger fue promovido
    fake_metrics = {"profit_factor_65": 1.5, "auc": 0.65, "n": 80}
    monkeypatch.setattr(rs, "_train_challenger", lambda v: fake_metrics)
    monkeypatch.setattr(rs, "_promote_champion", lambda v: None)
    monkeypatch.setattr(rs, "_profit_factor_on_recent", lambda s, n: None)  # sin champion
    monkeypatch.setattr(rs, "_cleanup_old_versions", lambda keep: None)

    with patch("retrain_scheduler.fetch_training_data", return_value=[{}] * 80):
        sched = RetrainScheduler()
        sched._run_retrain()

    assert sched._last_result is not None
    assert "promoted" in sched._last_result


def test_run_retrain_no_promote_when_challenger_worse(tmp_path, monkeypatch) -> None:
    """Si el challenger es peor que el champion → no se promueve."""
    import retrain_scheduler as rs

    monkeypatch.setattr(rs, "MODEL_DIR",      tmp_path)
    monkeypatch.setattr(rs, "VERSIONS_DIR",   tmp_path / "versions")
    monkeypatch.setattr(rs, "STATE_PATH",     tmp_path / "state.json")
    monkeypatch.setattr(rs, "CHAMPION_MODEL", tmp_path / "lgbm_model.pkl")
    monkeypatch.setattr(rs, "CHAMPION_CALIB", tmp_path / "platt_calibrator.pkl")

    fake_metrics = {"profit_factor_65": 0.9, "auc": 0.58, "n": 80}
    monkeypatch.setattr(rs, "_train_challenger",    lambda v: fake_metrics)
    monkeypatch.setattr(rs, "_promote_champion",    lambda v: None)
    monkeypatch.setattr(rs, "_profit_factor_on_recent", lambda s, n: 1.3)  # champion PF=1.3
    monkeypatch.setattr(rs, "_cleanup_old_versions", lambda keep: None)

    with patch("retrain_scheduler.fetch_training_data", return_value=[{}] * 80):
        sched = RetrainScheduler()
        sched._run_retrain()

    assert "rechazado" in sched._last_result


def test_run_retrain_no_action_when_train_fails(tmp_path, monkeypatch) -> None:
    """Si train_challenger devuelve None → sin promoción."""
    import retrain_scheduler as rs

    monkeypatch.setattr(rs, "MODEL_DIR",      tmp_path)
    monkeypatch.setattr(rs, "VERSIONS_DIR",   tmp_path / "versions")
    monkeypatch.setattr(rs, "STATE_PATH",     tmp_path / "state.json")
    monkeypatch.setattr(rs, "CHAMPION_MODEL", tmp_path / "lgbm_model.pkl")
    monkeypatch.setattr(rs, "CHAMPION_CALIB", tmp_path / "platt_calibrator.pkl")

    monkeypatch.setattr(rs, "_train_challenger", lambda v: None)

    with patch("retrain_scheduler.fetch_training_data", return_value=[{}] * 80):
        sched = RetrainScheduler()
        sched._run_retrain()

    assert sched._last_result is not None
    assert "rechazado" in sched._last_result


# ─── Scheduler lifecycle (sin asyncio real) ───────────────────────────────────

def test_scheduler_stop_when_not_running() -> None:
    """stop() sin start() previo no debe lanzar excepción."""
    sched = RetrainScheduler()
    sched.stop()   # no debe explotar


def test_scheduler_custom_interval() -> None:
    sched = RetrainScheduler(check_interval_h=12, min_new_trades=100)
    status = sched.get_status()
    assert status["check_interval_h"] == 12
    assert status["min_new_trades"]   == 100
