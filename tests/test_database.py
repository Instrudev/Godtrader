"""
tests/test_database.py – Tests unitarios para database.py

Usa bases de datos temporales (tmp_path de pytest) para no contaminar trades.db.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from database import (
    fetch_trades,
    get_winrate_by_hour,
    get_winrate_by_weekday,
    init_db,
    insert_trade,
    update_trade_result,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Base de datos temporal limpia para cada test."""
    path = tmp_path / "test_trades.db"
    init_db(path)
    return path


def _sample_trade(overrides: dict | None = None) -> dict:
    """Retorna un registro de trade mínimo válido."""
    base = {
        "timestamp":       "2026-04-20T10:00:00+00:00",
        "asset":           "EURUSD-OTC",
        "direction":       "CALL",
        "expiry_min":      2,
        "mode":            "paper",
        "price":           1.08500,
        "rsi":             28.5,
        "bb_pct_b":        0.12,
        "bb_width_pct":    0.185,
        "vol_rel":         1.0,
        "ema20":           1.08510,
        "ema200":          1.08450,
        "hour_utc":        10,
        "weekday":         0,
        "predicted_proba": 0.82,
        "ai_reasoning":    "test trade",
        "payout":          0.80,
        "open_price":      1.08500,
    }
    if overrides:
        base.update(overrides)
    return base


# ─── Tests de init_db ─────────────────────────────────────────────────────────

def test_init_db_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    assert not path.exists()
    init_db(path)
    assert path.exists()


def test_init_db_creates_trades_table(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    table_names = [t[0] for t in tables]
    assert "trades" in table_names


def test_init_db_idempotent(db: Path) -> None:
    """Llamar init_db dos veces no lanza error ni duplica tablas."""
    init_db(db)
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='trades'"
        ).fetchone()[0]
    assert count == 1


def test_init_db_all_columns_present(db: Path) -> None:
    expected_cols = {
        "id", "timestamp", "asset", "direction", "expiry_min", "mode",
        "price", "rsi", "bb_pct_b", "bb_width_pct", "vol_rel",
        "ema20", "ema200", "hour_utc", "weekday",
        "predicted_proba", "ai_reasoning", "order_id", "open_price", "payout",
        "result", "profit", "close_price", "pips_difference", "closed_at",
    }
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    assert expected_cols.issubset(cols), f"Columnas faltantes: {expected_cols - cols}"


# ─── Tests de insert_trade ────────────────────────────────────────────────────

def test_insert_trade_returns_id(db: Path) -> None:
    trade_id = insert_trade(_sample_trade(), path=db)
    assert isinstance(trade_id, int)
    assert trade_id >= 1


def test_insert_trade_multiple_ids_sequential(db: Path) -> None:
    id1 = insert_trade(_sample_trade(), path=db)
    id2 = insert_trade(_sample_trade(), path=db)
    assert id2 > id1


def test_insert_trade_record_persisted(db: Path) -> None:
    record = _sample_trade()
    trade_id = insert_trade(record, path=db)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    assert row is not None


def test_insert_trade_values_correct(db: Path) -> None:
    record = _sample_trade()
    trade_id = insert_trade(record, path=db)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone())
    assert row["asset"] == "EURUSD-OTC"
    assert row["direction"] == "CALL"
    assert row["rsi"] == pytest.approx(28.5)
    assert row["result"] == "PENDING"   # valor por defecto


def test_insert_trade_default_result_pending(db: Path) -> None:
    trade_id = insert_trade(_sample_trade(), path=db)
    with sqlite3.connect(db) as conn:
        result = conn.execute("SELECT result FROM trades WHERE id = ?", (trade_id,)).fetchone()[0]
    assert result == "PENDING"


# ─── Tests de update_trade_result ─────────────────────────────────────────────

def test_update_trade_result_win(db: Path) -> None:
    trade_id = insert_trade(_sample_trade(), path=db)
    update_trade_result(
        trade_id=trade_id,
        result="WIN",
        profit=0.80,
        payout=0.80,
        open_price=1.08500,
        close_price=1.08520,
        pips_difference=2.0,
        closed_at="2026-04-20T10:02:00+00:00",
        path=db,
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone())
    assert row["result"] == "WIN"
    assert row["profit"] == pytest.approx(0.80)
    assert row["close_price"] == pytest.approx(1.08520)
    assert row["pips_difference"] == pytest.approx(2.0)


def test_update_trade_result_loss(db: Path) -> None:
    trade_id = insert_trade(_sample_trade(), path=db)
    update_trade_result(
        trade_id=trade_id,
        result="LOSS",
        profit=-1.0,
        payout=0.80,
        open_price=1.08500,
        close_price=1.08480,
        pips_difference=2.0,
        closed_at="2026-04-20T10:02:00+00:00",
        path=db,
    )
    with sqlite3.connect(db) as conn:
        result = conn.execute("SELECT result FROM trades WHERE id = ?", (trade_id,)).fetchone()[0]
    assert result == "LOSS"


def test_update_nonexistent_trade_no_error(db: Path) -> None:
    """Actualizar un id que no existe no debe lanzar excepción."""
    update_trade_result(
        trade_id=999,
        result="WIN",
        profit=0.80,
        payout=0.80,
        open_price=1.0,
        close_price=1.1,
        pips_difference=1.0,
        closed_at="2026-04-20T10:02:00+00:00",
        path=db,
    )


# ─── Tests de fetch_trades ────────────────────────────────────────────────────

def test_fetch_trades_empty_db(db: Path) -> None:
    trades = fetch_trades(path=db)
    assert trades == []


def test_fetch_trades_returns_all(db: Path) -> None:
    insert_trade(_sample_trade({"mode": "paper"}), path=db)
    insert_trade(_sample_trade({"mode": "live"}), path=db)
    trades = fetch_trades(path=db)
    assert len(trades) == 2


def test_fetch_trades_filter_by_mode(db: Path) -> None:
    insert_trade(_sample_trade({"mode": "paper"}), path=db)
    insert_trade(_sample_trade({"mode": "live"}), path=db)
    insert_trade(_sample_trade({"mode": "backtest"}), path=db)
    paper_trades = fetch_trades(mode="paper", path=db)
    assert len(paper_trades) == 1
    assert paper_trades[0]["mode"] == "paper"


def test_fetch_trades_filter_by_asset(db: Path) -> None:
    insert_trade(_sample_trade({"asset": "EURUSD-OTC"}), path=db)
    insert_trade(_sample_trade({"asset": "GBPJPY-OTC"}), path=db)
    eur_trades = fetch_trades(asset="EURUSD-OTC", path=db)
    assert len(eur_trades) == 1
    assert eur_trades[0]["asset"] == "EURUSD-OTC"


def test_fetch_trades_returns_dicts(db: Path) -> None:
    insert_trade(_sample_trade(), path=db)
    trades = fetch_trades(path=db)
    assert isinstance(trades[0], dict)
    assert "asset" in trades[0]


# ─── Tests de winrate helpers ─────────────────────────────────────────────────

def _insert_closed_trade(db: Path, result: str, hour: int, weekday: int) -> None:
    trade_id = insert_trade(
        _sample_trade({"hour_utc": hour, "weekday": weekday}),
        path=db,
    )
    update_trade_result(
        trade_id=trade_id,
        result=result,
        profit=0.8 if result == "WIN" else -1.0,
        payout=0.80,
        open_price=1.0,
        close_price=1.1 if result == "WIN" else 0.9,
        pips_difference=1.0,
        closed_at="2026-04-20T10:02:00+00:00",
        path=db,
    )


def test_get_winrate_by_hour_empty(db: Path) -> None:
    result = get_winrate_by_hour(path=db)
    assert result == {}


def test_get_winrate_by_hour_minimum_trades(db: Path) -> None:
    """Debe requerir al menos 5 trades para incluir la hora."""
    for _ in range(4):
        _insert_closed_trade(db, "WIN", hour=10, weekday=0)
    result = get_winrate_by_hour(path=db)
    assert 10 not in result   # menos de 5 trades, no aparece


def test_get_winrate_by_hour_correct_calculation(db: Path) -> None:
    for _ in range(6):
        _insert_closed_trade(db, "WIN", hour=14, weekday=0)
    for _ in range(4):
        _insert_closed_trade(db, "LOSS", hour=14, weekday=0)
    result = get_winrate_by_hour(path=db)
    assert 14 in result
    assert result[14] == pytest.approx(0.60)


def test_get_winrate_by_weekday_correct(db: Path) -> None:
    for _ in range(3):
        _insert_closed_trade(db, "WIN", hour=10, weekday=1)
    for _ in range(2):
        _insert_closed_trade(db, "LOSS", hour=10, weekday=1)
    result = get_winrate_by_weekday(path=db)
    assert 1 in result
    assert result[1] == pytest.approx(0.60)
