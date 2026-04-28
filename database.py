"""
database.py – Capa de persistencia SQLite para el bot de trading.

Reemplaza trades_history.json con un esquema completo que registra:
- Features completas en el momento de la entrada (incluye features ML: Fase 4)
- Probabilidad predicha por el motor de decisión
- Resultado real, payout, pips de diferencia
- Metadatos de modo (live / paper / backtest)
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("trades.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT    NOT NULL,
    interval_s  INTEGER NOT NULL DEFAULT 60,  -- segundos por vela
    time        INTEGER NOT NULL,             -- Unix timestamp (apertura de la vela)
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL,
    fetched_at  TEXT    NOT NULL,             -- ISO-8601 UTC, momento de la descarga
    UNIQUE (asset, interval_s, time)
);

CREATE INDEX IF NOT EXISTS idx_candles_asset_time
    ON candles_history(asset, interval_s, time);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Identificación
    timestamp       TEXT    NOT NULL,          -- ISO-8601 UTC, momento de entrada
    asset           TEXT    NOT NULL,          -- e.g. "EURUSD-OTC"
    direction       TEXT    NOT NULL,          -- "CALL" o "PUT"
    expiry_min      INTEGER NOT NULL,          -- minutos de expiración
    mode            TEXT    NOT NULL,          -- "live" | "paper" | "backtest"
    -- Features en el momento de la entrada
    price           REAL,                      -- precio de cierre de la vela de entrada
    rsi             REAL,                      -- RSI(14)
    bb_pct_b        REAL,                      -- (price - bb_lower) / (bb_upper - bb_lower)
    bb_width_pct    REAL,                      -- (bb_upper - bb_lower) / bb_mid * 100
    vol_rel         REAL,                      -- volumen relativo (proxy en OTC)
    ema20           REAL,
    ema200          REAL,
    hour_utc        INTEGER,                   -- 0-23
    weekday         INTEGER,                   -- 0=lunes … 6=domingo
    -- Salida del motor de decisión
    predicted_proba REAL,                      -- pr / 100 (calibrado o raw LLM)
    ai_reasoning    TEXT,                      -- campo "an" del LLM / razón del ML
    -- Ejecución
    order_id        INTEGER,                   -- ID de la orden en el broker (-1 en paper)
    open_price      REAL,                      -- precio real de apertura de la orden
    payout          REAL,                      -- payout real del par en ese momento
    -- Resultado (se rellena al cierre)
    result          TEXT    DEFAULT 'PENDING', -- "WIN" | "LOSS" | "PENDING"
    profit          REAL,                      -- ganancia/pérdida neta en unidades de cuenta
    close_price     REAL,                      -- precio real al vencimiento
    pips_difference REAL,                      -- |open_price - close_price| en pips
    closed_at       TEXT,                      -- ISO-8601 UTC del cierre
    -- Features ML adicionales (Fase 4)
    streak_length   INTEGER,                   -- longitud de racha direccional actual
    streak_pct      REAL,                      -- percentil histórico de la racha
    ret_3           REAL,                      -- retorno acumulado últimas 3 velas
    ret_5           REAL,                      -- retorno acumulado últimas 5 velas
    ret_10          REAL,                      -- retorno acumulado últimas 10 velas
    autocorr_10     REAL,                      -- autocorrelación lag-1 de retornos (ventana 10)
    half_life       REAL,                      -- half-life OU estimada (velas)
    rel_atr         REAL                       -- ATR relativo a su media reciente
);

CREATE INDEX IF NOT EXISTS idx_trades_asset     ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_ts        ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_mode      ON trades(mode);
CREATE INDEX IF NOT EXISTS idx_trades_result    ON trades(result);
CREATE INDEX IF NOT EXISTS idx_trades_hour      ON trades(hour_utc);
CREATE INDEX IF NOT EXISTS idx_trades_weekday   ON trades(weekday);
"""


_ML_COLUMNS = [
    ("streak_length", "INTEGER"),
    ("streak_pct",    "REAL"),
    ("ret_3",         "REAL"),
    ("ret_5",         "REAL"),
    ("ret_10",        "REAL"),
    ("autocorr_10",   "REAL"),
    ("half_life",     "REAL"),
    ("rel_atr",       "REAL"),
    # Features PRNG (detección de sesgos del generador OTC)
    ("prng_last_digit_entropy",   "REAL"),
    ("prng_last_digit_mode_freq", "REAL"),
    ("prng_permutation_entropy",  "REAL"),
    ("prng_runs_test_z",          "REAL"),
    ("prng_transition_entropy",   "REAL"),
    ("prng_hurst_exponent",       "REAL"),
    ("prng_turning_point_ratio",  "REAL"),
    ("prng_autocorr_lag2",        "REAL"),
    ("prng_autocorr_lag5",        "REAL"),
    # Análisis post-trade (auto-aprendizaje de pérdidas)
    ("loss_type",            "TEXT"),    # "falsa_reversion"|"entrada_prematura"|"spread"|"tendencia_fuerte"
    ("price_after_5",        "REAL"),    # precio 5 velas después del cierre
    ("max_adverse_pips",     "REAL"),    # máximo movimiento en contra durante el trade
    ("max_favorable_pips",   "REAL"),    # máximo movimiento a favor durante el trade
]


def init_db(path: Path = DB_PATH) -> None:
    """Crea la base de datos y las tablas si no existen, y aplica migraciones."""
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
    logger.info(f"SQLite DB lista en {path.resolve()}")


def _migrate(conn: sqlite3.Connection) -> None:
    """Añade columnas ML que pueden faltar en bases de datos antiguas (idempotente)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    for col_name, col_type in _ML_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
            logger.debug(f"Migración: columna '{col_name}' añadida a trades")


def insert_trade(record: Dict[str, Any], path: Path = DB_PATH) -> int:
    """
    Inserta un nuevo registro de trade y devuelve su id.

    Args:
        record: Diccionario con los campos del trade (sin 'id').
        path:   Ruta al archivo de base de datos.

    Returns:
        El id asignado al nuevo registro.
    """
    cols = ", ".join(record.keys())
    placeholders = ", ".join("?" * len(record))
    sql = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
    with sqlite3.connect(path) as conn:
        cur = conn.execute(sql, list(record.values()))
        return cur.lastrowid


def update_trade_result(
    trade_id: int,
    result: str,
    profit: float,
    payout: float,
    open_price: float,
    close_price: float,
    pips_difference: float,
    closed_at: str,
    path: Path = DB_PATH,
) -> None:
    """Rellena los campos de resultado una vez que el trade cierra."""
    sql = """
        UPDATE trades
           SET result          = ?,
               profit          = ?,
               payout          = ?,
               open_price      = ?,
               close_price     = ?,
               pips_difference = ?,
               closed_at       = ?
         WHERE id = ?
    """
    with sqlite3.connect(path) as conn:
        conn.execute(
            sql,
            (result, profit, payout, open_price, close_price, pips_difference, closed_at, trade_id),
        )


def update_trade_result_simple(
    trade_id: int,
    result: str,
    profit: float,
    closed_at: str,
    path: Path = DB_PATH,
) -> None:
    """Actualiza únicamente resultado, profit y timestamp de cierre."""
    sql = "UPDATE trades SET result=?, profit=?, closed_at=? WHERE id=?"
    with sqlite3.connect(path) as conn:
        conn.execute(sql, (result, profit, closed_at, trade_id))


def update_trade_post_analysis(
    trade_id: int,
    loss_type: str,
    price_after_5: Optional[float] = None,
    max_adverse_pips: Optional[float] = None,
    max_favorable_pips: Optional[float] = None,
    path: Path = DB_PATH,
) -> None:
    """Rellena los campos de análisis post-trade para auto-aprendizaje."""
    sql = """
        UPDATE trades
           SET loss_type         = ?,
               price_after_5     = ?,
               max_adverse_pips  = ?,
               max_favorable_pips = ?
         WHERE id = ?
    """
    with sqlite3.connect(path) as conn:
        conn.execute(
            sql,
            (loss_type, price_after_5, max_adverse_pips, max_favorable_pips, trade_id),
        )


def fetch_recent_losses(
    days: int = 7,
    min_count: int = 3,
    path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Devuelve pérdidas recientes con análisis post-trade para el filtro anti-repetición.

    Args:
        days:      Últimos N días.
        min_count: Solo devuelve si hay al menos este número de pérdidas.

    Returns:
        Lista de dicts con los campos del trade (solo LOSS con loss_type no NULL).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sql = """
        SELECT * FROM trades
         WHERE result = 'LOSS'
           AND loss_type IS NOT NULL
           AND timestamp >= ?
         ORDER BY timestamp DESC
    """
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (cutoff,)).fetchall()
    losses = [dict(row) for row in rows]
    return losses if len(losses) >= min_count else []


def fetch_trades(
    mode: Optional[str] = None,
    asset: Optional[str] = None,
    limit: int = 1000,
    path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Devuelve trades del historial, opcionalmente filtrados por modo y/o activo.

    Args:
        mode:  Filtrar por "live", "paper" o "backtest". None = todos.
        asset: Filtrar por nombre de activo. None = todos.
        limit: Máximo de registros a devolver (orden descendente por timestamp).
        path:  Ruta al archivo de base de datos.

    Returns:
        Lista de dicts con todos los campos del trade.
    """
    conditions: List[str] = []
    params: List[Any] = []

    if mode:
        conditions.append("mode = ?")
        params.append(mode)
    if asset:
        conditions.append("asset = ?")
        params.append(asset)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_winrate_by_hour(asset: Optional[str] = None, path: Path = DB_PATH) -> Dict[int, float]:
    """
    Calcula el winrate histórico agrupado por hora UTC.

    Returns:
        Dict {hora_utc: winrate_float} para horas con al menos 5 trades cerrados.
    """
    conditions = ["result IN ('WIN','LOSS')"]
    params: List[Any] = []
    if asset:
        conditions.append("asset = ?")
        params.append(asset)

    sql = f"""
        SELECT hour_utc,
               COUNT(*) AS total,
               SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins
          FROM trades
         WHERE {' AND '.join(conditions)}
         GROUP BY hour_utc
        HAVING total >= 5
    """
    with sqlite3.connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {r[0]: r[2] / r[1] for r in rows}


def get_winrate_by_weekday(asset: Optional[str] = None, path: Path = DB_PATH) -> Dict[int, float]:
    """Calcula el winrate histórico agrupado por día de semana (0=lun … 6=dom)."""
    conditions = ["result IN ('WIN','LOSS')"]
    params: List[Any] = []
    if asset:
        conditions.append("asset = ?")
        params.append(asset)

    sql = f"""
        SELECT weekday,
               COUNT(*) AS total,
               SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins
          FROM trades
         WHERE {' AND '.join(conditions)}
         GROUP BY weekday
        HAVING total >= 5
    """
    with sqlite3.connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {r[0]: r[2] / r[1] for r in rows}


def save_candles(
    asset: str,
    candles: List[Dict[str, Any]],
    interval_s: int = 60,
    path: Path = DB_PATH,
) -> int:
    """
    Guarda velas históricas en candles_history.

    Usa INSERT OR REPLACE para que re-descargar el mismo activo actualice
    los datos sin duplicar filas (clave única: asset + interval_s + time).

    Args:
        asset:      Nombre del activo (e.g. "EURUSD-OTC").
        candles:    Lista de dicts con keys: time, open, high, low, close, [volume].
        interval_s: Duración de cada vela en segundos (default 60).
        path:       Ruta a la base de datos.

    Returns:
        Cantidad de filas insertadas o actualizadas.
    """
    if not candles:
        return 0

    from datetime import datetime, timezone
    fetched_at = datetime.now(timezone.utc).isoformat()

    rows = [
        (
            asset,
            interval_s,
            int(c["time"]),
            float(c["open"]),
            float(c["high"]),
            float(c["low"]),
            float(c["close"]),
            float(c["volume"]) if "volume" in c else None,
            fetched_at,
        )
        for c in candles
    ]

    sql = """
        INSERT OR REPLACE INTO candles_history
            (asset, interval_s, time, open, high, low, close, volume, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(path) as conn:
        conn.executemany(sql, rows)

    logger.debug(f"save_candles: {len(rows)} velas guardadas para {asset} ({interval_s}s)")
    return len(rows)


def load_candles(
    asset: str,
    interval_s: int = 60,
    limit: int = 500,
    path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Carga velas históricas desde candles_history ordenadas cronológicamente.

    Args:
        asset:      Nombre del activo.
        interval_s: Duración de cada vela en segundos.
        limit:      Máximo de velas a retornar (las más recientes).
        path:       Ruta a la base de datos.

    Returns:
        Lista de dicts {time, open, high, low, close, volume} ordenada ASC por time.
    """
    sql = """
        SELECT time, open, high, low, close, volume
          FROM candles_history
         WHERE asset = ? AND interval_s = ?
         ORDER BY time DESC
         LIMIT ?
    """
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (asset, interval_s, limit)).fetchall()

    # Revertir para que queden en orden cronológico (ASC)
    return [dict(r) for r in reversed(rows)]


def fetch_training_data(
    modes: Optional[List[str]] = None,
    path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Devuelve todos los trades cerrados (WIN/LOSS) aptos para entrenamiento ML.

    Ordena por timestamp ASC para preservar el orden temporal en walk-forward split.
    Solo incluye trades con resultado definido (no PENDING).

    Args:
        modes: Lista de modos a incluir (default: ["paper", "live"]).
        path:  Ruta a la base de datos.

    Returns:
        Lista de dicts ordenada por timestamp ASC.
    """
    if modes is None:
        modes = ["paper", "live"]

    placeholders = ",".join("?" * len(modes))
    sql = f"""
        SELECT *
          FROM trades
         WHERE result IN ('WIN', 'LOSS')
           AND mode IN ({placeholders})
         ORDER BY timestamp ASC
    """
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, modes).fetchall()
    return [dict(r) for r in rows]
