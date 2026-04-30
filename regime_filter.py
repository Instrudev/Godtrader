"""
regime_filter.py – Filtros de régimen para trading OTC (Exnova)

Cada filtro es una función pura que devuelve FilterResult(allow, filter_name, reason, auto_shutdown).
No se incluyen filtros de noticias macro ni sesiones forex: no aplican en OTC.

Filtros implementados:
  1. hour_profile_filter    – bloquea si winrate histórico del par en esa hora < umbral
  2. weekday_profile_filter – ídem por día de semana
  3. volatility_filter      – bloquea si ATR está en extremos (muy baja o muy alta)
  4. payout_filter          – bloquea si el payout actual es menor al mínimo aceptable
  5. daily_loss_filter      – apaga el bot si las pérdidas del día superan el límite
  6. consecutive_loss_filter– apaga el bot tras N pérdidas consecutivas
  7. max_trades_filter      – limita las operaciones diarias
  8. drift_filter           – bloquea si el generador OTC ha cambiado su algoritmo

Uso:
    from regime_filter import check_all_filters
    result = check_all_filters(df, asset, trade_log, amount, payout)
    if not result.allow:
        ...  # REGIME_BLOCK o BOT_AUTOSHUTDOWN
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from database import get_winrate_by_hour, get_winrate_by_weekday, fetch_recent_losses

logger = logging.getLogger(__name__)

# ─── Umbrales por defecto (todos configurables) ────────────────────────────────

MIN_WINRATE_HOURLY     = 0.52   # umbral restaurado (≥50 trades acumulados)
MIN_WINRATE_WEEKDAY    = 0.52   # umbral restaurado (≥50 trades acumulados)
MIN_WINRATE_DATA_FLOOR = 10     # trades mínimos históricos para aplicar filtro
ATR_PERIOD             = 14
ATR_PERCENTILE_LOW     = 30     # bloquear si ATR está por debajo de este percentil
ATR_PERCENTILE_HIGH    = 95     # bloquear si ATR está por encima de este percentil
MIN_PAYOUT             = 0.80   # payout mínimo del par para operar
MAX_DAILY_LOSSES       = 3      # pérdidas diarias máximas (auto-shutdown)
MAX_CONSECUTIVE_LOSSES = 3      # pérdidas consecutivas máximas (auto-shutdown)
MAX_TRADES_PER_DAY     = 15    # operaciones diarias máximas
BLOCKED_HOURS          = frozenset()                           # vacío — datos insuficientes (n<30 por hora), dejar que hour_profile_filter decida dinámicamente
BLOCKED_WEEKDAYS       = frozenset()                           # vacío — datos insuficientes, dejar que weekday_profile_filter decida dinámicamente
BLOCKED_ASSETS         = frozenset({"GBPUSD-OTC", "EURGBP-OTC", "AUDCAD-OTC", "GBPJPY-OTC"})
# Whitelist de subyacentes para opciones binarias — crypto/acciones/commodities dieron 0% WR
ALLOWED_ASSETS         = frozenset({
    "EURUSD-OTC", "USDJPY-OTC", "EURJPY-OTC", "USDCHF-OTC",
    "EURCHF-OTC", "AUDUSD-OTC", "NZDUSD-OTC", "USDCAD-OTC",
})
MIN_STREAK_LENGTH      = 3     # rachas de 1-2 velas no tienen edge
LOSS_PATTERN_DAYS      = 7     # días hacia atrás para buscar patrones de pérdida
LOSS_PATTERN_MIN_COUNT = 3     # mínimo de pérdidas similares para bloquear
LOSS_PATTERN_RSI_TOL   = 8.0   # tolerancia de RSI para considerar "similar"
LOSS_PATTERN_BB_TOL    = 0.15  # tolerancia de bb_pct_b para considerar "similar"


# ─── Resultado del filtro ─────────────────────────────────────────────────────

@dataclass
class FilterResult:
    allow: bool
    filter_name: str = ""
    reason: str = ""
    auto_shutdown: bool = False    # True → apagar bot, no solo saltear el ciclo

    @classmethod
    def ok(cls) -> "FilterResult":
        return cls(allow=True)

    @classmethod
    def block(cls, name: str, reason: str, shutdown: bool = False) -> "FilterResult":
        return cls(allow=False, filter_name=name, reason=reason, auto_shutdown=shutdown)


# ─── 1. Filtro de perfil horario ─────────────────────────────────────────────

def hour_profile_filter(
    hour_utc: int,
    asset: Optional[str] = None,
    min_winrate: float = MIN_WINRATE_HOURLY,
    data_floor: int = MIN_WINRATE_DATA_FLOOR,
) -> FilterResult:
    """
    Bloquea operaciones en franjas horarias donde el winrate histórico
    del activo sea menor al umbral mínimo.

    Pasa automáticamente si no hay suficiente data histórica (fail-open).
    """
    hourly = get_winrate_by_hour(asset=asset)
    if hour_utc not in hourly:
        return FilterResult.ok()   # sin datos suficientes → no bloquear

    wr = hourly[hour_utc]
    if wr < min_winrate:
        return FilterResult.block(
            "hour_profile_filter",
            f"Winrate en hora {hour_utc:02d}:00 UTC = {wr*100:.1f}% < mínimo {min_winrate*100:.0f}%",
        )
    return FilterResult.ok()


# ─── 2. Filtro de perfil por día de semana ────────────────────────────────────

def weekday_profile_filter(
    weekday: int,
    asset: Optional[str] = None,
    min_winrate: float = MIN_WINRATE_WEEKDAY,
) -> FilterResult:
    """
    Bloquea operaciones en días de semana con winrate histórico bajo.
    Aplica a sábado/domingo (OTC opera 24/7) donde el comportamiento suele diferir.

    weekday: 0=lunes … 6=domingo
    """
    by_day = get_winrate_by_weekday(asset=asset)
    if weekday not in by_day:
        return FilterResult.ok()

    wr = by_day[weekday]
    day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    name = day_names[weekday] if weekday < 7 else str(weekday)

    if wr < min_winrate:
        return FilterResult.block(
            "weekday_profile_filter",
            f"Winrate en {name} = {wr*100:.1f}% < mínimo {min_winrate*100:.0f}%",
        )
    return FilterResult.ok()


# ─── 3. Filtro de volatilidad (ATR relativo) ─────────────────────────────────

def volatility_filter(
    df: pd.DataFrame,
    atr_period: int = ATR_PERIOD,
    percentile_low: float = ATR_PERCENTILE_LOW,
    percentile_high: float = ATR_PERCENTILE_HIGH,
) -> FilterResult:
    """
    Bloquea cuando la volatilidad actual (ATR) está en los extremos de la
    distribución histórica de las últimas velas disponibles.

    - ATR < percentil 30 → mercado demasiado quieto (señal débil, spread relativo alto)
    - ATR > percentil 95 → volatilidad extrema (riesgo de manipulación / evento inusual)

    Usa las velas del DataFrame actual (≤ 300 velas ≈ 5h). En Fase 5, cuando haya
    datos acumulados, esto usará la ventana de 7 días real.
    """
    if len(df) < atr_period + 10:
        return FilterResult.ok()

    atr = _calculate_atr(df, atr_period)
    current_atr = float(atr.iloc[-1])

    if np.isnan(current_atr) or current_atr <= 0:
        return FilterResult.ok()

    atr_values = atr.dropna().values
    p_low  = float(np.percentile(atr_values, percentile_low))
    p_high = float(np.percentile(atr_values, percentile_high))

    if current_atr < p_low:
        return FilterResult.block(
            "volatility_filter",
            f"ATR actual ({current_atr:.5f}) < percentil {percentile_low} ({p_low:.5f}) → mercado demasiado quieto",
        )
    if current_atr > p_high:
        return FilterResult.block(
            "volatility_filter",
            f"ATR actual ({current_atr:.5f}) > percentil {percentile_high} ({p_high:.5f}) → volatilidad extrema",
        )
    return FilterResult.ok()


# ─── 4. Filtro de payout ──────────────────────────────────────────────────────

def payout_filter(
    current_payout: Optional[float],
    min_payout: float = MIN_PAYOUT,
) -> FilterResult:
    """
    Bloquea si el payout actual del par cae por debajo del umbral mínimo.
    Exnova reduce payouts dinámicamente; operar con payout bajo destruye el edge.

    Si current_payout es None (no se pudo obtener), pasa (fail-open).
    """
    if current_payout is None:
        return FilterResult.ok()   # sin dato de payout → no bloquear

    if current_payout < min_payout:
        return FilterResult.block(
            "payout_filter",
            f"Payout actual {current_payout*100:.0f}% < mínimo {min_payout*100:.0f}%",
        )
    return FilterResult.ok()


# ─── 5. Filtro de pérdida diaria ──────────────────────────────────────────────

def daily_loss_filter(
    trade_log: List[Dict],
    max_daily_losses: int = MAX_DAILY_LOSSES,
) -> FilterResult:
    """
    Apaga el bot si las pérdidas cerradas hoy superan el límite configurado.

    auto_shutdown=True: este filtro apaga el bot, no solo salta el ciclo.
    """
    today = date.today().isoformat()
    losses_today = sum(
        1
        for t in trade_log
        if t.get("result") == "LOSS"
        and str(t.get("timestamp", ""))[:10] == today
    )
    if losses_today >= max_daily_losses:
        return FilterResult.block(
            "daily_loss_filter",
            f"{losses_today} pérdidas hoy ≥ límite {max_daily_losses}. Bot detenido por el resto del día.",
            shutdown=True,
        )
    return FilterResult.ok()


# ─── 6. Filtro de pérdidas consecutivas ──────────────────────────────────────

def consecutive_loss_filter(
    trade_log: List[Dict],
    max_consecutive: int = MAX_CONSECUTIVE_LOSSES,
) -> FilterResult:
    """
    Apaga el bot tras N pérdidas consecutivas al final del trade_log.
    Una racha de pérdidas puede indicar drift del generador OTC.

    auto_shutdown=True: apaga el bot.
    """
    closed = [t for t in trade_log if t.get("result") in ("WIN", "LOSS")]
    if not closed:
        return FilterResult.ok()

    streak = 0
    for t in reversed(closed):
        if t["result"] == "LOSS":
            streak += 1
        else:
            break

    if streak >= max_consecutive:
        return FilterResult.block(
            "consecutive_loss_filter",
            f"{streak} pérdidas consecutivas ≥ límite {max_consecutive}. Posible drift del generador OTC.",
            shutdown=True,
        )
    return FilterResult.ok()


# ─── 7. Filtro de máximo de operaciones diarias ───────────────────────────────

def max_trades_filter(
    trade_log: List[Dict],
    max_trades: int = MAX_TRADES_PER_DAY,
) -> FilterResult:
    """
    Bloquea operaciones adicionales una vez alcanzado el límite diario.
    Controla la sobreexposición en un solo día.
    """
    today = date.today().isoformat()
    trades_today = sum(
        1
        for t in trade_log
        if str(t.get("timestamp", ""))[:10] == today
        and t.get("result") != "PENDING"
    )
    if trades_today >= max_trades:
        return FilterResult.block(
            "max_trades_filter",
            f"{trades_today} operaciones hoy ≥ límite {max_trades}. Esperando mañana.",
        )
    return FilterResult.ok()


# ─── 8. Filtro de drift del generador OTC ────────────────────────────────────

def drift_filter(asset: str) -> FilterResult:
    """
    Bloquea si generator_drift_detector ha detectado un cambio estadístico
    significativo en el feed OTC del activo en los últimos 7 días.

    Importación lazy para evitar ciclo de dependencia.
    """
    try:
        from generator_drift_detector import DriftDetector
        detector = DriftDetector.load()
        if detector.has_drift(asset, days=7):
            return FilterResult.block(
                "drift_filter",
                f"Drift estadístico detectado en {asset} en los últimos 7 días. "
                "Exnova probablemente cambió su algoritmo. NO operar.",
                shutdown=True,
            )
    except Exception as e:
        logger.debug(f"drift_filter: no se pudo consultar detector ({e}). Pasando.")
    return FilterResult.ok()


def blocked_hours_filter(hour: int) -> FilterResult:
    """Bloquea horas UTC con winrate históricamente < 40%."""
    if hour in BLOCKED_HOURS:
        return FilterResult.block(
            "blocked_hours_filter",
            f"Hora {hour:02d}h UTC bloqueada (winrate histórico < 40%)",
        )
    return FilterResult.ok()


def blocked_weekday_filter(weekday: int) -> FilterResult:
    """Bloquea días de la semana con winrate históricamente < 37%."""
    names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    if weekday in BLOCKED_WEEKDAYS:
        return FilterResult.block(
            "blocked_weekday_filter",
            f"{names[weekday]} bloqueado (winrate histórico < 37%)",
        )
    return FilterResult.ok()


def blocked_asset_filter(asset: str) -> FilterResult:
    """Bloquea activos en blacklist o fuera del whitelist de forex OTC."""
    if asset in BLOCKED_ASSETS:
        return FilterResult.block(
            "blocked_asset_filter",
            f"{asset} bloqueado (winrate histórico < 43%)",
        )
    if ALLOWED_ASSETS and asset not in ALLOWED_ASSETS:
        return FilterResult.block(
            "blocked_asset_filter",
            f"{asset} no está en whitelist de forex OTC",
        )
    return FilterResult.ok()


def min_streak_filter(df: pd.DataFrame) -> FilterResult:
    """Bloquea si la señal viene de una racha de 1-2 velas (sin edge demostrado)."""
    try:
        from indicators import get_streak_info
        streak = get_streak_info(df)
        length = streak.get("current_length", 0)
        if streak.get("is_extreme") and length < MIN_STREAK_LENGTH:
            return FilterResult.block(
                "min_streak_filter",
                f"Racha de {length} velas insuficiente (mínimo {MIN_STREAK_LENGTH})",
            )
    except Exception:
        pass
    return FilterResult.ok()


def loss_pattern_filter(
    df: pd.DataFrame,
    asset: str,
    direction: str,
    days: int = LOSS_PATTERN_DAYS,
    min_count: int = LOSS_PATTERN_MIN_COUNT,
    rsi_tol: float = LOSS_PATTERN_RSI_TOL,
    bb_tol: float = LOSS_PATTERN_BB_TOL,
) -> FilterResult:
    """
    Bloquea si las condiciones actuales son similares a pérdidas recientes recurrentes.

    Compara RSI, bb_pct_b, dirección, hora y tipo de pérdida actuales contra
    las pérdidas de los últimos N días. Si hay >= min_count pérdidas con
    condiciones similares, bloquea para no repetir el error.
    """
    try:
        losses = fetch_recent_losses(days=days, min_count=1)
    except Exception as e:
        logger.debug(f"loss_pattern_filter: error consultando pérdidas ({e}). Pasando.")
        return FilterResult.ok()

    if not losses:
        return FilterResult.ok()

    last = df.iloc[-1]
    current_rsi = float(last.get("rsi", 50.0))
    current_bb = float(last.get("bb_pct_b", 0.5) or 0.5)
    current_hour = int(last.get("hour_utc", 0) or 0)

    # Contar pérdidas con condiciones similares
    similar = []
    for loss in losses:
        # Misma dirección
        if loss.get("direction") != direction:
            continue
        # RSI similar (dentro de tolerancia)
        loss_rsi = loss.get("rsi")
        if loss_rsi is None or abs(float(loss_rsi) - current_rsi) > rsi_tol:
            continue
        # BB similar
        loss_bb = loss.get("bb_pct_b")
        if loss_bb is None or abs(float(loss_bb) - current_bb) > bb_tol:
            continue
        # Misma hora (±1)
        loss_hour = loss.get("hour_utc")
        if loss_hour is not None and abs(int(loss_hour) - current_hour) > 1:
            continue

        similar.append(loss)

    if len(similar) >= min_count:
        # Tipo de pérdida más frecuente
        types = [s.get("loss_type", "desconocido") for s in similar]
        most_common = max(set(types), key=types.count)

        return FilterResult.block(
            "loss_pattern_filter",
            f"Patrón de pérdida detectado: {len(similar)} pérdidas similares "
            f"en últimos {days}d ({direction} RSI≈{current_rsi:.0f} BB≈{current_bb:.2f} "
            f"h={current_hour}) | tipo frecuente: {most_common}",
        )

    return FilterResult.ok()


# ─── Función compuesta ────────────────────────────────────────────────────────

def check_all_filters(
    df: pd.DataFrame,
    asset: str,
    trade_log: List[Dict],
    payout: Optional[float] = None,
    direction: Optional[str] = None,
    min_winrate: float = MIN_WINRATE_HOURLY,
    min_payout: float = MIN_PAYOUT,
    max_daily_losses: int = MAX_DAILY_LOSSES,
    max_consecutive: int = MAX_CONSECUTIVE_LOSSES,
    max_trades: int = MAX_TRADES_PER_DAY,
) -> FilterResult:
    """
    Aplica todos los filtros en orden de menor a mayor coste computacional.
    Devuelve el primer FilterResult que bloquea, o FilterResult.ok() si todos pasan.

    Orden de evaluación (baratos primero):
      1. Hora bloqueada (instantáneo)
      2. Día bloqueado (instantáneo)
      3. Activo bloqueado (instantáneo)
      4. Pérdidas diarias (auto-shutdown, barato)
      5. Pérdidas consecutivas (auto-shutdown, barato)
      6. Máximo de operaciones diarias (barato)
      7. Drift del generador (archivo en disco, barato)
      8. Perfil horario (DB, medio)
      9. Perfil por día de semana (DB, medio)
     10. Volatilidad / ATR (cálculo numérico, medio)
     11. Payout (valor pasado como parámetro, barato)
     12. Racha mínima (cálculo, medio)
     13. Patrón de pérdida (DB, medio)
    """
    now_utc = datetime.now(timezone.utc)

    filters_to_run = [
        lambda: blocked_hours_filter(now_utc.hour),
        lambda: blocked_weekday_filter(now_utc.weekday()),
        lambda: blocked_asset_filter(asset),
        lambda: daily_loss_filter(trade_log, max_daily_losses),
        lambda: consecutive_loss_filter(trade_log, max_consecutive),
        lambda: max_trades_filter(trade_log, max_trades),
        lambda: drift_filter(asset),
        lambda: hour_profile_filter(now_utc.hour, asset, min_winrate),
        lambda: weekday_profile_filter(now_utc.weekday(), asset, min_winrate),
        lambda: volatility_filter(df),
        lambda: payout_filter(payout, min_payout),
        lambda: min_streak_filter(df),
        lambda: loss_pattern_filter(df, asset, direction) if direction else FilterResult.ok(),
    ]

    for run_filter in filters_to_run:
        result = run_filter()
        if not result.allow:
            return result

    return FilterResult.ok()


# ─── Helper: ATR(14) ─────────────────────────────────────────────────────────

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range usando suavizado EWM (consistente con el resto de indicadores).
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
