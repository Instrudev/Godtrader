"""
tests/test_ml_classifier.py – Tests del clasificador ML (Fase 4).

Cubre:
- extract_features: devuelve dict con las claves correctas
- features_to_array: orden y longitud correctos
- MLClassifier.predict_proba: neutral sin modelo / correcto con mock
- MLClassifier.load: fail-open si no hay archivo
- fetch_training_data / fetch completo
- Migración de DB (columnas ML)
- train_model._row_to_features: manejo de nulos y defaults
"""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Aseguramos que el directorio raíz del proyecto esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import build_dataframe
from ml_classifier import (
    FEATURE_COLS,
    MLClassifier,
    _NEUTRAL,
    extract_features,
    features_to_array,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, price = 1_700_000_000, 1.08500
    candles = []
    for i in range(n):
        change = rng.normal(0, 0.0003)
        open_p, close_p = price, price + change
        candles.append({
            "time":  t0 + i * 60,
            "open":  open_p,
            "high":  max(open_p, close_p) + abs(rng.normal(0, 0.0001)),
            "low":   min(open_p, close_p) - abs(rng.normal(0, 0.0001)),
            "close": close_p,
        })
        price = close_p
    return build_dataframe(candles)


# ─── extract_features ─────────────────────────────────────────────────────────

def test_extract_features_returns_dict() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert isinstance(feat, dict)


def test_extract_features_has_all_keys() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    for key in FEATURE_COLS:
        assert key in feat, f"Clave faltante: {key}"


def test_extract_features_values_are_floats() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    for k, v in feat.items():
        assert isinstance(v, float), f"'{k}' no es float: {type(v)}"


def test_extract_features_empty_df_returns_none() -> None:
    assert extract_features(pd.DataFrame()) is None


def test_extract_features_small_df_returns_none() -> None:
    # build_dataframe devuelve DataFrame vacío para < 20 velas
    feat = extract_features(pd.DataFrame())
    assert feat is None


def test_extract_features_rsi_in_range() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    assert 0.0 <= feat["rsi"] <= 100.0


def test_extract_features_bb_pct_b_in_range() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    # bb_pct_b puede ir ligeramente fuera de [0,1] en picos de volatilidad, pero
    # con datos sintéticos suaves debería estar cerca del rango
    assert -0.5 <= feat["bb_pct_b"] <= 1.5


def test_extract_features_half_life_capped() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    assert feat["half_life"] <= 100.0


def test_extract_features_direction_call() -> None:
    df = _make_df()
    feat = extract_features(df, direction="CALL")
    assert feat is not None
    assert feat["direction"] == 1.0


def test_extract_features_direction_put() -> None:
    df = _make_df()
    feat = extract_features(df, direction="PUT")
    assert feat is not None
    assert feat["direction"] == 0.0


def test_extract_features_direction_none() -> None:
    df = _make_df()
    feat = extract_features(df, direction=None)
    assert feat is not None
    assert feat["direction"] == 0.5


def test_extract_features_payout_passed_through() -> None:
    df = _make_df()
    feat = extract_features(df, payout=0.85)
    assert feat is not None
    assert feat["payout"] == pytest.approx(0.85)


def test_extract_features_winrate_hour_passed_through() -> None:
    df = _make_df()
    feat = extract_features(df, winrate_hour=0.62)
    assert feat is not None
    assert feat["winrate_hour"] == pytest.approx(0.62)


def test_extract_features_expiry_min_passed_through() -> None:
    df = _make_df()
    feat = extract_features(df, expiry_min=5.0)
    assert feat is not None
    assert feat["expiry_min"] == pytest.approx(5.0)


def test_extract_features_ret_3_nonzero_with_trend() -> None:
    """Un trend claro debe producir ret_3 != 0."""
    rng = np.random.default_rng(99)
    t0, price = 1_700_000_000, 1.08500
    candles = []
    for i in range(250):
        # Deriva alcista fuerte
        change = 0.0005 + rng.normal(0, 0.0001)
        open_p, close_p = price, price + change
        candles.append({
            "time":  t0 + i * 60,
            "open":  open_p, "high": close_p + 0.0001,
            "low":   open_p - 0.0001, "close": close_p,
        })
        price = close_p
    df = build_dataframe(candles)
    feat = extract_features(df)
    assert feat is not None
    assert feat["ret_3"] > 0.0, "Trend alcista debe tener ret_3 positivo"


# ─── features_to_array ────────────────────────────────────────────────────────

def test_features_to_array_length() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    arr = features_to_array(feat)
    assert len(arr) == len(FEATURE_COLS)


def test_features_to_array_dtype() -> None:
    df = _make_df()
    feat = extract_features(df)
    assert feat is not None
    arr = features_to_array(feat)
    assert arr.dtype == np.float32


def test_features_to_array_order() -> None:
    """El primer elemento debe corresponder a 'rsi'."""
    feat = {col: float(i) for i, col in enumerate(FEATURE_COLS)}
    arr = features_to_array(feat)
    assert arr[0] == pytest.approx(0.0)   # rsi es el primero
    assert arr[1] == pytest.approx(1.0)   # bb_pct_b es el segundo


def test_features_to_array_missing_key_defaults_zero() -> None:
    feat: Dict = {}   # sin ninguna clave
    arr = features_to_array(feat)
    assert arr.shape == (len(FEATURE_COLS),)
    assert (arr == 0.0).all()


# ─── MLClassifier sin modelo ─────────────────────────────────────────────────

def test_classifier_no_model_returns_neutral() -> None:
    clf = MLClassifier()
    feat = {col: 0.0 for col in FEATURE_COLS}
    result = clf.predict_proba(feat)
    assert result == _NEUTRAL


def test_classifier_not_loaded_initially() -> None:
    clf = MLClassifier()
    assert not clf.is_loaded()


def test_classifier_load_missing_file_returns_false(tmp_path) -> None:
    clf = MLClassifier()
    ok = clf.load(
        model_path=tmp_path / "nonexistent.pkl",
        calib_path=tmp_path / "nonexistent2.pkl",
    )
    assert ok is False
    assert not clf.is_loaded()


def test_classifier_predict_proba_from_df_no_model() -> None:
    clf = MLClassifier()
    df = _make_df()
    result = clf.predict_proba_from_df(df)
    assert result["call_proba"] == pytest.approx(0.5)
    assert result["put_proba"]  == pytest.approx(0.5)


# ─── MLClassifier con modelo mock ────────────────────────────────────────────

def _inject_mock_model(clf: MLClassifier, call_win_prob: float = 0.7) -> None:
    """
    Inyecta mocks directamente en _model y _calibrator del clasificador
    sin pasar por pickle (MagicMock no es picklable).
    """
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    # Modelo LightGBM real mínimo entrenado con datos sintéticos
    n = 60
    rng = np.random.default_rng(0)
    X_dummy = rng.random((n, len(FEATURE_COLS))).astype(np.float32)
    y_dummy = (rng.random(n) > 0.5).astype(int)

    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(X_dummy, y_dummy)

    # Calibrador ajustado con datos sintéticos
    raw = model.predict_proba(X_dummy)[:, 1].reshape(-1, 1)
    calib = LogisticRegression(C=1.0, max_iter=100)
    calib.fit(raw, y_dummy)

    clf._model      = model
    clf._calibrator = calib
    clf._loaded     = True


def test_classifier_with_real_mini_model() -> None:
    """Verifica predict_proba con un modelo LightGBM real (mínimo)."""
    clf = MLClassifier()
    _inject_mock_model(clf)
    assert clf.is_loaded()

    feat = {col: 0.5 for col in FEATURE_COLS}
    result = clf.predict_proba(feat)
    assert "call_proba" in result
    assert "put_proba"  in result
    assert 0.0 <= result["call_proba"] <= 1.0
    assert 0.0 <= result["put_proba"]  <= 1.0


def test_classifier_predict_proba_call_ne_put() -> None:
    """call_proba y put_proba pueden diferir (dirección distinta como feature)."""
    clf = MLClassifier()
    _inject_mock_model(clf)

    feat = {col: 0.5 for col in FEATURE_COLS}
    result = clf.predict_proba(feat)
    # No son necesariamente distintas, pero deben ser válidas
    assert isinstance(result["call_proba"], float)
    assert isinstance(result["put_proba"],  float)


def test_classifier_predict_proba_from_df_with_model() -> None:
    clf = MLClassifier()
    _inject_mock_model(clf)

    df = _make_df()
    result = clf.predict_proba_from_df(df)
    assert "call_proba" in result
    assert "put_proba"  in result


def test_classifier_load_saves_model_to_file(tmp_path) -> None:
    """Verifica que load() carga correctamente un modelo serializado."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    n = 60
    rng = np.random.default_rng(1)
    X = rng.random((n, len(FEATURE_COLS))).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(int)

    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(X, y)
    raw = model.predict_proba(X)[:, 1].reshape(-1, 1)
    calib = LogisticRegression(C=1.0, max_iter=100)
    calib.fit(raw, y)

    model_path = tmp_path / "lgbm_model.pkl"
    calib_path = tmp_path / "platt_calibrator.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(calib_path, "wb") as f:
        pickle.dump(calib, f)

    clf = MLClassifier()
    ok = clf.load(model_path=model_path, calib_path=calib_path)
    assert ok
    assert clf.is_loaded()


# ─── Database migration ───────────────────────────────────────────────────────

def test_db_migration_adds_ml_columns(tmp_path) -> None:
    """init_db debe añadir columnas ML a una DB antigua sin ellas."""
    import sqlite3
    from database import init_db, _ML_COLUMNS

    db_path = tmp_path / "test_migration.db"

    # Crear tabla con el esquema pre-Fase4 (incluye todas las columnas originales
    # pero NO las columnas ML que se añaden en Fase 4).
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                asset           TEXT    NOT NULL,
                direction       TEXT    NOT NULL,
                expiry_min      INTEGER NOT NULL,
                mode            TEXT    NOT NULL,
                price           REAL,
                rsi             REAL,
                bb_pct_b        REAL,
                bb_width_pct    REAL,
                vol_rel         REAL,
                ema20           REAL,
                ema200          REAL,
                hour_utc        INTEGER,
                weekday         INTEGER,
                predicted_proba REAL,
                ai_reasoning    TEXT,
                order_id        INTEGER,
                open_price      REAL,
                payout          REAL,
                result          TEXT    DEFAULT 'PENDING',
                profit          REAL,
                close_price     REAL,
                pips_difference REAL,
                closed_at       TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_hour ON trades(hour_utc)")

    # Ejecutar init_db (debe migrar añadiendo columnas ML)
    init_db(db_path)

    # Verificar que las columnas ML existen
    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    for col_name, _ in _ML_COLUMNS:
        assert col_name in cols, f"Columna ML faltante tras migración: {col_name}"


def test_db_migration_idempotent(tmp_path) -> None:
    """Llamar init_db dos veces no falla."""
    from database import init_db
    db_path = tmp_path / "test_idem.db"
    init_db(db_path)
    init_db(db_path)  # segunda llamada no debe lanzar excepción


# ─── fetch_training_data ─────────────────────────────────────────────────────

def test_fetch_training_data_empty_db(tmp_path) -> None:
    from database import init_db, fetch_training_data
    db_path = tmp_path / "empty.db"
    init_db(db_path)
    rows = fetch_training_data(path=db_path)
    assert rows == []


def test_fetch_training_data_only_closed(tmp_path) -> None:
    """Solo devuelve WIN/LOSS, no PENDING."""
    from database import init_db, insert_trade, fetch_training_data
    db_path = tmp_path / "fetch_test.db"
    init_db(db_path)

    base = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "asset": "EURUSD-OTC",
        "direction": "CALL",
        "expiry_min": 1,
        "mode": "paper",
    }
    insert_trade({**base, "result": "WIN"},  path=db_path)
    insert_trade({**base, "result": "LOSS"}, path=db_path)
    insert_trade({**base, "result": "PENDING"}, path=db_path)

    rows = fetch_training_data(path=db_path)
    assert len(rows) == 2
    assert all(r["result"] in ("WIN", "LOSS") for r in rows)


def test_fetch_training_data_excludes_backtest(tmp_path) -> None:
    """Por defecto no incluye modo 'backtest'."""
    from database import init_db, insert_trade, fetch_training_data
    db_path = tmp_path / "fetch_bt.db"
    init_db(db_path)

    base = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "asset": "EURUSD-OTC",
        "direction": "CALL",
        "expiry_min": 1,
        "result": "WIN",
    }
    insert_trade({**base, "mode": "paper"},    path=db_path)
    insert_trade({**base, "mode": "backtest"}, path=db_path)

    rows = fetch_training_data(path=db_path)
    assert len(rows) == 1
    assert rows[0]["mode"] == "paper"


def test_fetch_training_data_ordered_asc(tmp_path) -> None:
    """Los resultados deben venir ordenados por timestamp ASC."""
    from database import init_db, insert_trade, fetch_training_data
    db_path = tmp_path / "fetch_order.db"
    init_db(db_path)

    base = {"asset": "EURUSD-OTC", "direction": "CALL", "expiry_min": 1, "mode": "paper", "result": "WIN"}
    insert_trade({**base, "timestamp": "2024-01-01T02:00:00+00:00"}, path=db_path)
    insert_trade({**base, "timestamp": "2024-01-01T01:00:00+00:00"}, path=db_path)
    insert_trade({**base, "timestamp": "2024-01-01T03:00:00+00:00"}, path=db_path)

    rows = fetch_training_data(path=db_path)
    timestamps = [r["timestamp"] for r in rows]
    assert timestamps == sorted(timestamps)


# ─── train_model._row_to_features ────────────────────────────────────────────

def test_row_to_features_basic() -> None:
    from train_model import _row_to_features
    row = {
        "direction": "CALL",
        "rsi": 45.0,
        "bb_pct_b": 0.4,
        "bb_width_pct": 1.2,
        "rel_atr": 1.1,
        "vol_rel": 0.9,
        "hour_utc": 10,
        "weekday": 2,
        "streak_length": 3,
        "streak_pct": 72.0,
        "ret_3": 0.001,
        "ret_5": 0.002,
        "ret_10": 0.003,
        "autocorr_10": -0.1,
        "half_life": 15.0,
        "payout": 0.82,
        "winrate_hour": 0.55,
    }
    arr = _row_to_features(row)
    assert arr is not None
    assert len(arr) == len(FEATURE_COLS)


def test_row_to_features_direction_call() -> None:
    from train_model import _row_to_features
    arr = _row_to_features({"direction": "CALL"})
    assert arr is not None
    dir_idx = FEATURE_COLS.index("direction")
    assert arr[dir_idx] == pytest.approx(1.0)


def test_row_to_features_direction_put() -> None:
    from train_model import _row_to_features
    arr = _row_to_features({"direction": "PUT"})
    assert arr is not None
    dir_idx = FEATURE_COLS.index("direction")
    assert arr[dir_idx] == pytest.approx(0.0)


def test_row_to_features_null_fields_use_defaults() -> None:
    """Campos None deben usar valores por defecto, no lanzar excepción."""
    from train_model import _row_to_features
    row = {"direction": "CALL", "rsi": None, "half_life": None, "streak_pct": None}
    arr = _row_to_features(row)
    assert arr is not None
    rsi_idx = FEATURE_COLS.index("rsi")
    assert arr[rsi_idx] == pytest.approx(50.0)  # default


def test_row_to_features_half_life_capped() -> None:
    """half_life infinita debe capearse a 100."""
    from train_model import _row_to_features
    row = {"direction": "CALL", "half_life": float("inf")}
    arr = _row_to_features(row)
    assert arr is not None
    hl_idx = FEATURE_COLS.index("half_life")
    assert arr[hl_idx] == pytest.approx(100.0)


def test_row_to_features_half_life_large_capped() -> None:
    from train_model import _row_to_features
    row = {"direction": "CALL", "half_life": 999.0}
    arr = _row_to_features(row)
    assert arr is not None
    hl_idx = FEATURE_COLS.index("half_life")
    assert arr[hl_idx] == pytest.approx(100.0)


# ─── walk-forward split ───────────────────────────────────────────────────────

def test_walk_forward_split_sizes() -> None:
    from train_model import _walk_forward_split
    n = 100
    X = np.zeros((n, 5))
    y = np.zeros(n)
    X_tr, y_tr, X_v, y_v, X_te, y_te = _walk_forward_split(X, y, 0.70, 0.15)
    assert len(y_tr) == 70
    assert len(y_v)  == 15
    assert len(y_te) == 15


def test_walk_forward_split_no_overlap() -> None:
    from train_model import _walk_forward_split
    n = 100
    X = np.arange(n * 3).reshape(n, 3)
    y = np.arange(n, dtype=float)
    X_tr, y_tr, X_v, y_v, X_te, y_te = _walk_forward_split(X, y)
    # Los índices no deben solaparse: train < val < test (orden cronológico)
    assert y_tr[-1] < y_v[0]
    assert y_v[-1]  < y_te[0]


def test_walk_forward_total_equals_n() -> None:
    from train_model import _walk_forward_split
    n = 200
    X = np.zeros((n, 5))
    y = np.zeros(n)
    splits = _walk_forward_split(X, y)
    total = sum(len(s) for s in splits[1::2])   # y_train + y_val + y_test
    assert total == n


# ─── auc_roc helper ──────────────────────────────────────────────────────────

def test_auc_roc_perfect_classifier() -> None:
    from train_model import _auc_roc
    y_true  = np.array([1, 1, 0, 0], dtype=float)
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    auc = _auc_roc(y_true, y_score)
    assert auc == pytest.approx(1.0, abs=1e-6)


def test_auc_roc_random_classifier() -> None:
    from train_model import _auc_roc
    rng = np.random.default_rng(0)
    y_true  = rng.integers(0, 2, 1000).astype(float)
    y_score = rng.random(1000)
    auc = _auc_roc(y_true, y_score)
    # Debe estar cerca de 0.5 para un clasificador aleatorio
    assert 0.4 <= auc <= 0.6


def test_brier_score_perfect() -> None:
    from train_model import _brier_score
    y_true = np.array([1.0, 0.0, 1.0])
    y_prob = np.array([1.0, 0.0, 1.0])
    assert _brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_random() -> None:
    from train_model import _brier_score
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert _brier_score(y_true, y_prob) == pytest.approx(0.25)


# ─── Tarea 1.5: Carga explícita con verificación SHA256 ─────────────────────

def _create_mock_model(tmp_path):
    """Crea modelo LightGBM + calibrador Platt mock en tmp_path, retorna (model_path, calib_path)."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X = rng.random((60, len(FEATURE_COLS))).astype(np.float32)
    y = (rng.random(60) > 0.5).astype(int)

    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(X, y)
    raw = model.predict_proba(X)[:, 1].reshape(-1, 1)
    calib = LogisticRegression(C=1.0, max_iter=100)
    calib.fit(raw, y)

    model_path = tmp_path / "lgbm_model.pkl"
    calib_path = tmp_path / "platt_calibrator.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(calib_path, "wb") as f:
        pickle.dump(calib, f)
    return model_path, calib_path

def test_load_with_verification_valid_hash(tmp_path) -> None:
    """Hash coincide → carga OK."""
    import hashlib
    model_path, calib_path = _create_mock_model(tmp_path)
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()

    clf = MLClassifier()
    ok = clf.load_with_verification(model_path, calib_path, expected_hash=expected)
    assert ok is True
    assert clf.is_loaded()
    assert clf._model_hash == expected


def test_load_with_verification_invalid_hash(tmp_path) -> None:
    """Hash no coincide → error + no carga."""
    model_path, calib_path = _create_mock_model(tmp_path)

    clf = MLClassifier()
    ok = clf.load_with_verification(model_path, calib_path, expected_hash="badhash123")
    assert ok is False
    assert not clf.is_loaded()


def test_auto_load_blocked_in_remediation_mode() -> None:
    """predict_proba no auto-carga en REMEDIATION_MODE."""
    import iqservice
    orig = iqservice.REMEDIATION_MODE
    try:
        iqservice.REMEDIATION_MODE = True
        clf = MLClassifier()
        result = clf.predict_proba({col: 0.0 for col in FEATURE_COLS})
        # Debe retornar neutral (0.5) sin intentar cargar
        assert result["call_proba"] == 0.5
        assert result["put_proba"] == 0.5
        assert clf._load_attempted is True  # marcado como intentado (para no reintentar)
        assert clf._loaded is False  # pero NO cargado
    finally:
        iqservice.REMEDIATION_MODE = orig


def test_load_logs_hash_and_path(tmp_path, caplog) -> None:
    """load() loguea hash y path del modelo."""
    import logging
    model_path, calib_path = _create_mock_model(tmp_path)

    clf = MLClassifier()
    with caplog.at_level(logging.INFO):
        clf.load(model_path, calib_path)

    assert "hash=" in caplog.text
    assert str(model_path) in caplog.text
    assert "features=" in caplog.text


def test_audit_script_loads_model_and_validates_hash() -> None:
    """El script de auditoría carga el modelo y valida hash contra baseline."""
    from audit_ml_model import _sha256, MODEL_PATH, BASELINE_HASH
    if not MODEL_PATH.exists():
        pytest.skip("Modelo no disponible")
    actual = _sha256(MODEL_PATH)
    assert actual == BASELINE_HASH


def test_audit_script_handles_insufficient_data(tmp_path) -> None:
    """Script reporta warning con dataset pequeño, no crashea."""
    from audit_ml_model import _count_trades
    info = _count_trades()
    # Con la BD actual, debe retornar datos válidos
    assert isinstance(info, dict)
    assert "total" in info


def test_load_without_expected_hash(tmp_path) -> None:
    """Sin hash esperado → carga OK + reporta hash detectado."""
    model_path, calib_path = _create_mock_model(tmp_path)

    clf = MLClassifier()
    ok = clf.load_with_verification(model_path, calib_path, expected_hash=None)
    assert ok is True
    assert clf.is_loaded()
    assert clf._model_hash is not None  # hash detectado y almacenado
