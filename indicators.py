"""
indicators.py – Motor de Análisis Técnico v3 (OTC Edition)

Funciones clásicas (EMA, RSI, BB, VolRel, S/R, patrones de vela) mantenidas
para compatibilidad con ai_brain.py y el pipeline existente.

Nuevas funciones OTC (Fase 3):
  - calculate_atr()         → ATR(14) como pd.Series
  - calculate_relative_atr()→ ATR actual / media ATR de la ventana (ratio)
  - estimate_half_life()    → Ornstein-Uhlenbeck: half-life en velas M1
  - get_streak_info()       → racha direccional actual vs distribución histórica
  - otc_signal()            → señal OTC compuesta: dirección + confianza + razón
  - load_strategy_config()  → carga strategy_config.yaml
  - pre_qualify()           → PRE-CAL reescrita para OTC (primario: racha+OU)
  - pre_qualify_classical() → lógica clásica RSI+BB (fallback / compatibilidad)

Sin dependencias TA externas: solo numpy, pandas, yaml (PyYAML).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("strategy_config.yaml")


# ─── Constructor Principal del DataFrame ──────────────────────────────────────

def build_dataframe(candles: List[Dict]) -> pd.DataFrame:
    """
    Convierte la lista cruda de velas de IQ Option en un DataFrame enriquecido
    con todos los indicadores técnicos pre-calculados.

    Requiere ≥ 205 velas para que EMA200 y RSI14 sean estadísticamente fiables.
    El volumen se incluye si está presente en los datos; si no, se usa un proxy
    basado en el rango de la vela (high - low) × una constante.
    """
    if not candles or len(candles) < 20:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    # Volumen: usar campo real si existe, o proxy de rango como fallback
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float).replace(0, np.nan)
    else:
        # Proxy de liquidez: rango de vela normalizado (suficiente para RelVol)
        df["volume"] = (df["high"] - df["low"]).astype(float)

    closes = df["close"]

    # ── Indicadores ──────────────────────────────────────────────────────────
    df["ema20"]    = _ema(closes, 20)
    df["ema200"]   = _ema(closes, 200)
    df["rsi"]      = _rsi(closes, 14)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bollinger(closes, 14, 2.0)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    df["vol_rel"]  = _relative_volume(df["volume"], period=20)

    # ── Indicadores OTC adicionales (Fase 3) ──────────────────────────────────
    df["atr"]     = calculate_atr(df, period=14)
    atr_mean      = df["atr"].dropna().mean()
    df["rel_atr"] = (df["atr"] / atr_mean).fillna(1.0) if atr_mean > 1e-10 else 1.0

    return df


# ─── Indicadores Numéricos ────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Media Móvil Exponencial (suavizado Wilder vía EWM de pandas)."""
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI con suavizado de Wilder (alpha = 1/period).
    Retorna [0, 100]. Fallback a 50 cuando el histórico es insuficiente.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _bollinger(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger. Retorna (superior, media, inferior)."""
    mid   = series.rolling(window=period).mean()
    std   = series.rolling(window=period).std(ddof=0)
    return mid + std_dev * std, mid, mid - std_dev * std


def _relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Volumen Relativo = volumen_actual / media_volumen(N periodos).
    > 1.5 → volumen significativamente alto (movimiento con convicción).
    < 0.5 → mercado inusualmente quieto (señal débil, evitar operar).
    """
    avg = volume.rolling(window=period, min_periods=1).mean()
    return (volume / avg.replace(0, np.nan)).fillna(1.0)


# ─── Soporte y Resistencia ────────────────────────────────────────────────────

def find_support_resistance(
    df: pd.DataFrame,
    swing_window: int = 5,
    cluster_pct: float = 0.0015,
    n_levels: int = 3,
) -> Dict:
    """
    Detecta niveles S/R a partir de extremos locales en las 300 velas.

    Retorna:
        {'resistances': [r1, r2, ...], 'supports': [s1, s2, ...]}
    """
    if len(df) < swing_window * 2 + 1:
        return {"resistances": [], "supports": []}

    current_price = float(df["close"].iloc[-1])
    raw_res = _find_local_maxima(df["high"].values, swing_window)
    raw_sup = _find_local_minima(df["low"].values, swing_window)

    resistances = sorted(
        r for r in _cluster_levels(raw_res, cluster_pct) if r > current_price
    )[:n_levels]
    supports = sorted(
        (s for s in _cluster_levels(raw_sup, cluster_pct) if s < current_price),
        reverse=True,
    )[:n_levels]

    return {"resistances": resistances, "supports": supports}


def _find_local_maxima(values: np.ndarray, window: int) -> List[float]:
    return [
        float(values[i])
        for i in range(window, len(values) - window)
        if values[i] >= max(values[i - window : i + window + 1])
    ]


def _find_local_minima(values: np.ndarray, window: int) -> List[float]:
    return [
        float(values[i])
        for i in range(window, len(values) - window)
        if values[i] <= min(values[i - window : i + window + 1])
    ]


def _cluster_levels(levels: List[float], threshold_pct: float) -> List[float]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters: List[List[float]] = [[levels[0]]]
    for level in levels[1:]:
        ref = clusters[-1][0]
        if ref > 0 and (level - ref) / ref < threshold_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [float(np.mean(c)) for c in clusters]

# ─── Módulo de Memoria de Ciclos (Microestructura) ────────────────────────────

def calculate_cycle_stats(df: pd.DataFrame) -> Dict:
    """
    Calcula estadísticas reales de las últimas 180 velas (3h) para entender 
    el comportamiento empírico del algoritmo del broker.
    """
    if len(df) < 50:
        return {"rsi_respect_rate": 0, "bb_breakout_avg": 0, "market_state": "UNKNOWN", "rsi_touches": 0}

    # Recortar a las últimas 180 velas (máximo)
    recent_df = df.iloc[-180:].reset_index(drop=True)
    
    # 1. Tasa de respeto del RSI
    rsi_touches = 0
    rsi_respects = 0
    
    for i in range(len(recent_df) - 2):
        row = recent_df.iloc[i]
        rsi = row["rsi"]
        
        if rsi < 35: # Sobrevendida, debe rebotar al alza
            rsi_touches += 1
            if recent_df.iloc[i+1]["close"] > row["close"] or recent_df.iloc[i+2]["close"] > row["close"]:
                rsi_respects += 1
        elif rsi > 65: # Sobrecomprada, debe rebotar a la baja
            rsi_touches += 1
            if recent_df.iloc[i+1]["close"] < row["close"] or recent_df.iloc[i+2]["close"] < row["close"]:
                rsi_respects += 1
                
    rsi_rate = round((rsi_respects / rsi_touches * 100), 1) if rsi_touches > 0 else 50.0

    # 2. Distancia de Ruptura BB (Pips que se aleja antes de revertir)
    breakout_distances = []
    for i in range(len(recent_df)):
        c = float(recent_df.iloc[i]["close"])
        bb_u = float(recent_df.iloc[i]["bb_upper"])
        bb_l = float(recent_df.iloc[i]["bb_lower"])
        if c > bb_u:
            breakout_distances.append(abs(c - bb_u))
        elif c < bb_l:
            breakout_distances.append(abs(bb_l - c))
            
    bb_avg_break = np.mean(breakout_distances) if breakout_distances else 0.0

    # 3. Estado del Mercado
    outside_bb = len(breakout_distances)
    above_ema200 = sum(recent_df["close"] > recent_df["ema200"])
    trend_intensity = abs((above_ema200 / len(recent_df)) - 0.5) * 2 # 0 a 1
    
    if trend_intensity > 0.7 or outside_bb > 25:
        market_state = "Tendencia Infinita Fuerte"
    elif outside_bb < 10 and trend_intensity < 0.3:
        market_state = "Rango Errático Lateral"
    else:
        market_state = "Transición / Estructura Normal"

    return {
        "rsi_respect_rate": rsi_rate,
        "bb_breakout_avg": round(bb_avg_break, 5),
        "market_state": market_state,
        "rsi_touches": rsi_touches
    }


# ─── Detección de Patrones de Vela ───────────────────────────────────────────

def detect_patterns(df: pd.DataFrame, lookback: int = 10) -> List[Dict]:
    """
    Identifica patrones de reversión institucionales en las últimas N velas.

    Patrones detectados:
    - HAMMER:        Mecha inferior larga → señal alcista de reversión
    - SHOOTING_STAR: Mecha superior larga → señal bajista de reversión
    - BULL_ENGULF:   Vela alcista engloba vela bajista previa
    - BEAR_ENGULF:   Vela bajista engloba vela alcista previa
    - DOJI:          Cuerpo mínimo → indecisión, posible reversión

    Retorna lista de dicts con patrón, timestamp y dirección implícita.
    """
    recent = df.tail(lookback).reset_index(drop=True)
    found  = []

    for i in range(1, len(recent)):
        curr = recent.iloc[i]
        prev = recent.iloc[i - 1]

        body        = abs(curr["close"] - curr["open"])
        total_range = curr["high"] - curr["low"]
        if total_range < 1e-8:
            continue

        upper_wick = curr["high"] - max(curr["close"], curr["open"])
        lower_wick = min(curr["close"], curr["open"]) - curr["low"]
        body_ratio = body / total_range
        t_str      = curr["time"].strftime("%H:%M")

        # ── Pin Bars ──────────────────────────────────────────────────────────
        if lower_wick >= 2 * body and body_ratio < 0.35:
            found.append({"pattern": "HAMMER", "time": t_str, "bias": "BULLISH"})

        if upper_wick >= 2 * body and body_ratio < 0.35:
            found.append({"pattern": "SHOOTING_STAR", "time": t_str, "bias": "BEARISH"})

        # ── Doji ──────────────────────────────────────────────────────────────
        if body_ratio < 0.10:
            found.append({"pattern": "DOJI", "time": t_str, "bias": "REVERSAL"})

        # ── Velas Envolventes ─────────────────────────────────────────────────
        prev_body = abs(prev["close"] - prev["open"])
        if prev_body < 1e-8:
            continue

        bull_engulf = (
            curr["close"] > curr["open"]   # alcista actual
            and prev["close"] < prev["open"]  # bajista anterior
            and curr["open"] <= prev["close"]
            and curr["close"] >= prev["open"]
        )
        bear_engulf = (
            curr["close"] < curr["open"]   # bajista actual
            and prev["close"] > prev["open"]  # alcista anterior
            and curr["open"] >= prev["close"]
            and curr["close"] <= prev["open"]
        )

        if bull_engulf:
            found.append({"pattern": "BULL_ENGULF", "time": t_str, "bias": "BULLISH"})
        if bear_engulf:
            found.append({"pattern": "BEAR_ENGULF", "time": t_str, "bias": "BEARISH"})

    return found


# ─── Filtro de Pre-Calificación (OTC v3) ──────────────────────────────────────

def pre_qualify(df: pd.DataFrame, asset: str = "", config: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Filtro de "despertar" para la IA — reescrito para OTC (Fase 3).

    Lógica primaria (OTC):
      1. Racha direccional actual supera el percentil mínimo histórico → señal de reversión
      2. ATR relativo en rango aceptable (no mercado muerto ni extremo)

    Lógica secundaria / fallback (clásica):
      Si no hay racha extrema y enable_classical_fallback=true:
      RSI en zona extrema + precio en zona exterior de BB

    Args:
        df:     DataFrame con indicadores calculados por build_dataframe().
        asset:  Nombre del activo (para cargar perfil del config).
        config: Override del strategy_config. Si None, carga desde YAML.

    Returns:
        (True, razón) si califica para análisis IA
        (False, razón) si debe omitirse el ciclo
    """
    if len(df) < 30:
        return False, "DataFrame insuficiente para análisis OTC"

    cfg = config if config is not None else _get_asset_config(asset)

    # ── 1. ATR relativo (check rápido de volatilidad local) ───────────────────
    rel_atr = float(df["rel_atr"].iloc[-1]) if "rel_atr" in df.columns else calculate_relative_atr(df)
    rel_atr_min = cfg.get("rel_atr_min", 0.40)
    rel_atr_max = cfg.get("rel_atr_max", 2.50)

    if rel_atr < rel_atr_min:
        return False, f"ATR relativo {rel_atr:.2f} < {rel_atr_min} (mercado demasiado quieto)"
    if rel_atr > rel_atr_max:
        return False, f"ATR relativo {rel_atr:.2f} > {rel_atr_max} (volatilidad extrema)"

    # ── 2. Señal primaria OTC: racha extrema → reversión esperada ─────────────
    streak = get_streak_info(df)
    pct_min = cfg.get("streak_percentile_min", 90)

    if streak["percentile"] >= pct_min and streak["current_length"] >= 4:
        direction = "CALL" if streak["direction"] == "bear" else "PUT"
        # Half-life obligatorio: mean-reversion debe ser demostrable
        hl = estimate_half_life(df)
        if hl > 25:
            return False, (
                f"Racha {streak['direction'].upper()} {streak['current_length']}v "
                f"(pct={streak['percentile']:.0f}%) pero half-life {hl:.1f}v > 25 "
                "(mean-reversion insuficiente)"
            )
        hl_str = f" | Half-life≈{hl:.1f}v"
        reason = (
            f"Racha {streak['direction'].upper()} {streak['current_length']}v "
            f"(pct={streak['percentile']:.0f}%) → reversión {direction}"
            f"{hl_str} | ATRrel={rel_atr:.2f}"
        )
        # Filtro secundario: price action confirmatorio (opcional)
        if cfg.get("require_price_action", False):
            patterns = detect_patterns(df, lookback=3)
            aligned = [
                p for p in patterns
                if (direction == "CALL" and p["bias"] == "BULLISH")
                or (direction == "PUT"  and p["bias"] == "BEARISH")
            ]
            if not aligned:
                return False, f"Price action requerida pero no encontrada para {direction}"
        return True, reason

    # ── 3. Señal BB 2-Candle Reversal (2 velas fuera de BB + RSI + reversión) ─
    bb2_ok, bb2_dir, bb2_reason = detect_bb_two_candle_reversal(df)
    if bb2_ok:
        return True, bb2_reason

    # ── 4. Señal BB Body Reversal (1 vela, legacy) ───────────────────────────
    reversal_ok, reversal_reason = detect_bb_body_reversal(df)
    if reversal_ok:
        return True, reversal_reason

    # ── 5. Fallback clásico: RSI + BB ─────────────────────────────────────────
    if not cfg.get("enable_classical_fallback", True):
        return False, f"Sin racha extrema (pct={streak['percentile']:.0f}% < {pct_min}%)"

    return pre_qualify_classical(df)


def pre_qualify_classical(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Lógica original de pre-calificación: RSI extremo + toque de BB.
    Mantenida como fallback y para compatibilidad con ai_brain.py.
    """
    last     = df.iloc[-1]
    rsi      = float(last["rsi"])
    price    = float(last["close"])
    bb_upper = float(last["bb_upper"])
    bb_lower = float(last["bb_lower"])
    vol_rel  = float(last.get("vol_rel", 1.0))

    rsi_extreme = rsi < 25 or rsi > 75
    bb_range    = bb_upper - bb_lower
    pct_b       = (price - bb_lower) / bb_range if bb_range > 1e-10 else 0.5
    bb_touch    = pct_b <= 0.10 or pct_b >= 0.90

    if rsi_extreme and bb_touch:
        rsi_label = "SOBREVENDIDO" if rsi < 25 else "SOBRECOMPRADO"
        bb_label  = "ZONA_INF" if pct_b <= 0.10 else "ZONA_SUP"
        return True, f"[clásico] RSI {rsi:.1f} ({rsi_label}) + {bb_label} | VolRel={vol_rel:.2f}x"

    reasons = []
    if not rsi_extreme:
        reasons.append(f"RSI {rsi:.1f} [neutral 25-75]")
    if not bb_touch:
        reasons.append(f"precio central (pct_b={pct_b:.2f})")
    return False, " / ".join(reasons)


def detect_bb_two_candle_reversal(df: pd.DataFrame) -> Tuple[bool, Optional[str], str]:
    """
    Señal de reversión basada en 2 velas consecutivas fuera de las Bandas de Bollinger.

    CALL (rebote alcista):
      1. Las últimas 2 velas tienen > 50% de su cuerpo por debajo de BB lower
      2. RSI < 20 (sobreventa)
      3. Última vela es alcista (close > open) — confirmación de rebote

    PUT (rebote bajista):
      1. Las últimas 2 velas tienen > 50% de su cuerpo por encima de BB upper
      2. RSI > 80 (sobrecompra)
      3. Última vela es bajista (close < open) — confirmación de caída

    Returns:
        (True, "CALL"/"PUT", razón) si hay señal
        (False, None, razón) si no califica
    """
    if len(df) < 22:
        return False, None, "DataFrame insuficiente para análisis de 2 velas + BB"

    c1 = df.iloc[-2]  # penúltima vela
    c2 = df.iloc[-1]  # última vela

    def _body_below_bb_lower_pct(candle) -> float:
        """Qué porcentaje del cuerpo de la vela está por debajo de BB lower."""
        o, c = float(candle["open"]), float(candle["close"])
        bb_low = float(candle["bb_lower"])
        body_top = max(o, c)
        body_bot = min(o, c)
        body_size = body_top - body_bot
        if body_size < 1e-10:
            return 0.0
        # Porción del cuerpo que cae debajo de bb_lower
        below = max(0.0, bb_low - body_bot)
        return min(below / body_size, 1.0)

    def _body_above_bb_upper_pct(candle) -> float:
        """Qué porcentaje del cuerpo de la vela está por encima de BB upper."""
        o, c = float(candle["open"]), float(candle["close"])
        bb_up = float(candle["bb_upper"])
        body_top = max(o, c)
        body_bot = min(o, c)
        body_size = body_top - body_bot
        if body_size < 1e-10:
            return 0.0
        # Porción del cuerpo que sobresale por encima de bb_upper
        above = max(0.0, body_top - bb_up)
        return min(above / body_size, 1.0)

    rsi = float(c2["rsi"])
    c2_open = float(c2["open"])
    c2_close = float(c2["close"])

    # ── Check CALL: 2 velas > 70% body debajo de BB lower + RSI oversold + divergencia + vela alcista
    pct_below_1 = _body_below_bb_lower_pct(c1)
    pct_below_2 = _body_below_bb_lower_pct(c2)

    if pct_below_1 > 0.70 and pct_below_2 > 0.70:
        if rsi < 20:
            # Divergencia RSI alcista: precio hace lower low pero RSI sube
            if len(df) >= 4:
                low_N  = float(c2["low"])
                low_N3 = float(df.iloc[-4]["low"])
                rsi_N3 = float(df.iloc[-4]["rsi"])
                if not (low_N < low_N3 and rsi > rsi_N3):
                    return False, None, (
                        f"Sin divergencia RSI alcista (low={low_N:.5f} vs low_N3={low_N3:.5f}, "
                        f"rsi={rsi:.1f} vs rsi_N3={rsi_N3:.1f})"
                    )
            if c2_close > c2_open:  # vela alcista = confirmación de rebote
                return True, "CALL", (
                    f"[bb_2candle] CALL: 2 velas con >{pct_below_1:.0%}/{pct_below_2:.0%} "
                    f"body bajo BB-Lower | RSI={rsi:.1f} (sobreventa) | "
                    f"divergencia RSI confirmada | última vela alcista"
                )
            return False, None, f"RSI {rsi:.1f} OK pero última vela no es alcista (sin confirmación)"
        return False, None, f"2 velas bajo BB-Lower pero RSI {rsi:.1f} no confirma sobreventa (<20)"

    # ── Check PUT: 2 velas > 70% body encima de BB upper + RSI overbought + divergencia + vela bajista
    pct_above_1 = _body_above_bb_upper_pct(c1)
    pct_above_2 = _body_above_bb_upper_pct(c2)

    if pct_above_1 > 0.70 and pct_above_2 > 0.70:
        if rsi > 80:
            # Divergencia RSI bajista: precio hace higher high pero RSI baja
            if len(df) >= 4:
                high_N  = float(c2["high"])
                high_N3 = float(df.iloc[-4]["high"])
                rsi_N3  = float(df.iloc[-4]["rsi"])
                if not (high_N > high_N3 and rsi < rsi_N3):
                    return False, None, (
                        f"Sin divergencia RSI bajista (high={high_N:.5f} vs high_N3={high_N3:.5f}, "
                        f"rsi={rsi:.1f} vs rsi_N3={rsi_N3:.1f})"
                    )
            if c2_close < c2_open:  # vela bajista = confirmación de caída
                return True, "PUT", (
                    f"[bb_2candle] PUT: 2 velas con >{pct_above_1:.0%}/{pct_above_2:.0%} "
                    f"body sobre BB-Upper | RSI={rsi:.1f} (sobrecompra) | "
                    f"divergencia RSI confirmada | última vela bajista"
                )
            return False, None, f"RSI {rsi:.1f} OK pero última vela no es bajista (sin confirmación)"
        return False, None, f"2 velas sobre BB-Upper pero RSI {rsi:.1f} no confirma sobrecompra (>80)"

    return False, None, "Sin patrón de 2 velas fuera de BB"


def detect_bb_body_reversal(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Señal PUT de reversión: vela verde con bb_mid en el 10% inferior del cuerpo
    (o por debajo). Indica que el precio se extendió sobre la media → presión bajista.

    Condición principal (Opción A):
        close > open  (vela verde)
        bb_mid ≤ open + (close - open) * 0.10

    Filtros de precisión:
        RSI > 55        → confirma sobreextensión real
        bb_width > 0.3% → volatilidad suficiente (excluye squeeze)
        vol_rel ≥ 1.0   → el movimiento alcista tiene volumen
    """
    if len(df) < 20:
        return False, "DataFrame insuficiente"

    last     = df.iloc[-1]
    open_    = float(last["open"])
    close    = float(last["close"])
    bb_mid   = float(last["bb_mid"])
    bb_width = float(last["bb_width"])
    rsi      = float(last["rsi"])
    vol_rel  = float(last.get("vol_rel", 1.0))

    body = close - open_
    if body <= 0:
        return False, "Vela no es alcista"

    threshold = open_ + body * 0.10
    if bb_mid > threshold:
        return False, f"bb_mid al {(bb_mid - open_) / body * 100:.0f}% del cuerpo (necesita ≤10%)"

    if rsi <= 55:
        return False, f"RSI {rsi:.1f} ≤ 55 (sin sobreextensión)"

    if bb_width <= 0.30:
        return False, f"BB-Width {bb_width:.3f}% ≤ 0.30% (squeeze)"

    if vol_rel < 1.0:
        return False, f"VolRel {vol_rel:.2f}x < 1.0 (volumen insuficiente)"

    ext_pct = (threshold - bb_mid) / body * 100
    return True, (
        f"[bb_body_reversal] Vela verde extendida sobre BB-Mid "
        f"(ext={ext_pct:.0f}% sobre umbral) | RSI={rsi:.1f} | "
        f"BB-W={bb_width:.3f}% | VolRel={vol_rel:.2f}x"
    )


# ─── Mantenimiento de Mercado y Caza de Liquidez ──────────────────────────────

def calculate_adherence_index(df: pd.DataFrame, window: int = 30) -> str:
    """
    Índice de Adherencia (Backtesting en vivo).
    Analiza las últimas 'window' velas buscando señales teóricas.
    """
    if len(df) < window + 5:
        return "NEUTRAL"
        
    recent = df.iloc[-(window+3):].reset_index(drop=True)
    wins = 0
    total = 0
    
    for i in range(len(recent) - 3):
        row = recent.iloc[i]
        rsi = float(row["rsi"])
        close_t3 = float(recent.iloc[i+3]["close"])
        close_t0 = float(row["close"])
        
        # Simulamos entradas de 3 mins
        if rsi < 35 or close_t0 <= float(row["bb_lower"]):
            total += 1
            if close_t3 > close_t0: wins += 1
        elif rsi > 65 or close_t0 >= float(row["bb_upper"]):
            total += 1
            if close_t3 < close_t0: wins += 1
            
    if total < 3: 
        return "NEUTRAL"
        
    win_rate = (wins / total) * 100
    if win_rate > 65: return "EFICIENTE (Generoso)"
    if win_rate < 40: return "CAZADOR (Manipulación)"
    return f"NEUTRAL ({win_rate:.1f}%)"

def detect_vsa_anomaly(df: pd.DataFrame) -> bool:
    """
    Análisis VSA (Esfuerzo vs Resultado).
    Retorna True si hay divergencia Volumen/Spread.
    """
    if len(df) < 20: 
        return False
    
    last = df.iloc[-1]
    # mean body
    recent_bodies = abs(df["close"].iloc[-20:] - df["open"].iloc[-20:])
    mean_body = float(recent_bodies.mean())
    if mean_body < 1e-8: mean_body = 1e-8
    
    body = float(abs(last["close"] - last["open"]))
    vol_rel = float(last["vol_rel"])
    
    # Falsa Expansión / Fakeout M5
    if body > (mean_body * 1.5) and vol_rel < 0.8:
        return True
    
    # Absorción / Pared Institucional Algorítmica
    if body < (mean_body * 0.5) and vol_rel > 1.8:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# INDICADORES OTC — Fase 3
# Detectan regularidades del generador de precios del broker, no de mercado real.
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range usando suavizado EWM.
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)

    Función pública exportada para uso en regime_filter y backtester.
    """
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev  = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calculate_relative_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Ratio ATR_actual / media_ATR_ventana.

    > 1.0 → volatilidad por encima de su media reciente
    < 1.0 → mercado más quieto de lo habitual
    """
    if "atr" in df.columns:
        atr = df["atr"]
    else:
        atr = calculate_atr(df, period)

    current = float(atr.iloc[-1])
    mean    = float(atr.dropna().mean())
    if mean < 1e-10 or not np.isfinite(current):
        return 1.0
    return current / mean


def estimate_half_life(df: pd.DataFrame, window: int = 100) -> float:
    """
    Estima la half-life de mean reversion del proceso de precios mediante
    regresión Ornstein-Uhlenbeck (OLS):

        ΔX_t = α + β·X_{t-1} + ε_t

    Half-life = ln(2) / |β|  (solo válido si β < 0)

    Args:
        df:     DataFrame con columna 'close'.
        window: Número de velas a usar (últimas N velas, default=100 ≈ 1.7h M1).

    Returns:
        Half-life en velas. float('inf') si no hay mean reversion (β ≥ 0).
    """
    if df.empty or "close" not in df.columns:
        return float("inf")
    prices = df["close"].iloc[-window:].astype(float).values
    if len(prices) < 20:
        return float("inf")

    delta = np.diff(prices)
    lag   = prices[:-1]

    # OLS: delta = alpha + beta * lag
    X = np.column_stack([np.ones_like(lag), lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
        beta = float(coeffs[1])
    except Exception:
        return float("inf")

    if beta >= 0 or not np.isfinite(beta):
        return float("inf")   # sin mean reversion o proceso explosivo

    hl = float(np.log(2) / abs(beta))
    return hl if np.isfinite(hl) else float("inf")


def get_streak_info(df: pd.DataFrame, history_window: int = 200) -> Dict:
    """
    Analiza la racha direccional actual versus la distribución histórica.

    Una racha es una secuencia consecutiva de velas alcistas (close > open)
    o bajistas (close < open). Cuando la racha actual supera el percentil 75
    histórico, la probabilidad de reversión es estadísticamente alta.

    Args:
        df:             DataFrame con columnas 'open' y 'close'.
        history_window: Número de velas históricas para calcular la distribución.

    Returns:
        Dict con:
          direction:       'bull', 'bear' o 'none'
          current_length:  longitud de la racha actual (velas)
          percentile:      percentil de la racha actual vs distribución histórica (0-100)
          is_extreme:      True si percentile >= 75
          mean_streak:     longitud media de rachas históricas
          p90_streak:      percentil 90 de rachas históricas
    """
    if len(df) < 10:
        return _empty_streak()

    # Tolerancia para dojis (velas grises donde open y close son casi idénticos)
    body   = df["close"].astype(float) - df["open"].astype(float)
    # Considerar 0 (doji) si el cuerpo es menor a 1e-6 para evitar diferencias por flotantes mínimos
    dirs   = np.where(np.abs(body.values) <= 1e-6, 0, np.sign(body.values))

    # ── Distribución histórica de rachas ──────────────────────────────────────
    hist   = dirs[-history_window:]
    all_streaks = _measure_all_streaks(hist)

    # ── Racha actual (desde el final) ─────────────────────────────────────────
    cur_dir    = None
    cur_length = 0
    for d in reversed(dirs):
        if d == 0:
            break  # Un doji rompe la racha consecutiva en lugar de saltarlo
        if cur_dir is None:
            cur_dir = d
        if d == cur_dir:
            cur_length += 1
        else:
            break

    if cur_dir is None or cur_length == 0 or not all_streaks:
        return _empty_streak()

    sorted_s = np.sort(all_streaks)
    pct      = float(np.searchsorted(sorted_s, cur_length, side="right") / len(sorted_s) * 100)

    return {
        "direction":      "bull" if cur_dir > 0 else "bear",
        "current_length": int(cur_length),
        "percentile":     pct,
        "is_extreme":     pct >= 75,
        "mean_streak":    float(np.mean(all_streaks)),
        "p90_streak":     float(np.percentile(all_streaks, 90)) if len(all_streaks) > 5 else 0.0,
    }


def otc_signal(df: pd.DataFrame, asset: str = "", config: Optional[Dict] = None) -> Dict:
    """
    Señal OTC compuesta que combina racha, half-life y ATR relativo.

    Devuelve un dict con:
      has_signal:    bool — hay señal accionable
      direction:     'CALL', 'PUT' o None
      confidence:    float [0, 1] — confianza en la señal
      reason:        str — descripción legible
      streak:        dict — info de la racha
      half_life:     float — half-life en velas
      rel_atr:       float — ATR relativo
    """
    cfg = config if config is not None else _get_asset_config(asset)

    streak  = get_streak_info(df)
    hl      = estimate_half_life(df)
    rel_atr = calculate_relative_atr(df)
    pct_min = cfg.get("streak_percentile_min", 70)

    # Sin señal base
    result: Dict[str, Any] = {
        "has_signal": False,
        "direction":  None,
        "confidence": 0.0,
        "reason":     "Sin condiciones de entrada",
        "streak":     streak,
        "half_life":  hl,
        "rel_atr":    rel_atr,
    }

    if not streak["is_extreme"] or streak["percentile"] < pct_min:
        result["reason"] = (
            f"Racha {streak['direction']} {streak['current_length']}v "
            f"(pct={streak['percentile']:.0f}% < {pct_min}%)"
        )
        return result

    rel_min = cfg.get("rel_atr_min", 0.40)
    rel_max = cfg.get("rel_atr_max", 2.50)
    if rel_atr < rel_min or rel_atr > rel_max:
        result["reason"] = f"ATR relativo {rel_atr:.2f} fuera de rango [{rel_min}, {rel_max}]"
        return result

    direction = "CALL" if streak["direction"] == "bear" else "PUT"

    # Confianza: función del percentil de la racha y de la half-life
    # percentil 75 → 0.78 de confianza; percentil 95 → ~0.92; hl corta → bonus
    base_conf = 0.70 + (streak["percentile"] - 75) / 100 * 0.20
    hl_bonus  = 0.05 if hl <= cfg.get("half_life_max_candles", 25) else 0.0
    confidence = min(base_conf + hl_bonus, 0.95)

    hl_str = f"{hl:.1f}v" if hl < float("inf") else "∞"
    result.update({
        "has_signal": True,
        "direction":  direction,
        "confidence": confidence,
        "reason": (
            f"Racha {streak['direction'].upper()} {streak['current_length']}v "
            f"(pct={streak['percentile']:.0f}%) → {direction} | "
            f"HL={hl_str} | ATRrel={rel_atr:.2f}"
        ),
    })
    return result


# ─── Config ───────────────────────────────────────────────────────────────────

def load_strategy_config(path: Path = _CONFIG_PATH) -> Dict:
    """
    Carga strategy_config.yaml. Usa yaml si está disponible, json como fallback.
    Devuelve dict vacío si el archivo no existe.
    """
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json
        # Intentar como JSON (YAML es superset de JSON)
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Error cargando {path}: {e}")
    return {}


def _get_asset_config(asset: str) -> Dict:
    """
    Construye la configuración efectiva para un activo combinando:
      1. default del YAML
      2. overrides del activo (si existe)
      3. overrides de la franja horaria actual (si corresponde)
    """
    from datetime import datetime, timezone
    full_cfg = load_strategy_config()
    cfg = dict(full_cfg.get("default", {}))

    # Override por activo
    asset_overrides = full_cfg.get("assets", {}).get(asset, {})
    cfg.update(asset_overrides)

    # Override por franja horaria
    hour_now = datetime.now(timezone.utc).hour
    weekday  = datetime.now(timezone.utc).weekday()  # 0=lun, 5=sab, 6=dom
    for profile in full_cfg.get("time_profiles", []):
        if hour_now in profile.get("hours", []):
            # El perfil weekend solo aplica sábado/domingo
            if profile.get("name") == "weekend" and weekday < 5:
                continue
            time_overrides = {k: v for k, v in profile.items() if k not in ("name", "hours")}
            cfg.update(time_overrides)
            break

    return cfg


# ─── Helpers privados OTC ─────────────────────────────────────────────────────

def _measure_all_streaks(dirs: np.ndarray) -> List[int]:
    """
    Mide la longitud de todas las rachas en un array de direcciones (+1/-1/0).
    Un doji (0) rompe la racha actual.
    """
    streaks: List[int] = []
    current_dir: Optional[int] = None
    current_len = 0

    for d in dirs:
        if d == 0:
            if current_len > 0:
                streaks.append(current_len)
            current_dir = None
            current_len = 0
            continue
            
        if current_dir is None:
            current_dir = d
            current_len = 1
        elif d == current_dir:
            current_len += 1
        else:
            streaks.append(current_len)
            current_dir = d
            current_len = 1

    if current_len > 0:
        streaks.append(current_len)

    return streaks


def _empty_streak() -> Dict:
    return {
        "direction":      "none",
        "current_length": 0,
        "percentile":     0.0,
        "is_extreme":     False,
        "mean_streak":    0.0,
        "p90_streak":     0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES PRNG — Detección de sesgos en generadores de precios OTC
# Diseñadas para explotar debilidades de PRNG (LCG, Mersenne Twister, etc.),
# no para análisis técnico de mercados reales.
# ═══════════════════════════════════════════════════════════════════════════════


def prng_last_digit_entropy(closes: np.ndarray, window: int = 60, decimals: int = 5) -> float:
    """
    Entropía normalizada del último dígito significativo del precio.

    Los PRNG tipo LCG tienen bits bajos menos aleatorios que los altos.
    Una distribución perfectamente uniforme da entropía = 1.0.
    Valores < 1.0 indican sesgo en los dígitos finales.
    """
    prices = closes[-window:]
    if len(prices) < 10:
        return 1.0
    last_digits = np.array([int(round(p * 10**decimals)) % 10 for p in prices])
    counts = np.bincount(last_digits, minlength=10).astype(float)
    total = counts.sum()
    if total < 1:
        return 1.0
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(10))  # normalizado [0, 1]


def prng_last_digit_mode_freq(closes: np.ndarray, window: int = 60, decimals: int = 5) -> float:
    """
    Frecuencia del dígito final más común.

    Esperado ~0.10 para distribución uniforme. Valores > 0.15 indican
    sesgo directo en el generador.
    """
    prices = closes[-window:]
    if len(prices) < 10:
        return 0.10
    last_digits = np.array([int(round(p * 10**decimals)) % 10 for p in prices])
    counts = np.bincount(last_digits, minlength=10)
    return float(counts.max() / len(last_digits))


def prng_permutation_entropy(closes: np.ndarray, m: int = 3, delay: int = 1, window: int = 60) -> float:
    """
    Entropía de permutación: mide la complejidad ordinal de la serie.

    Para cada sub-secuencia de longitud m, calcula su patrón ordinal
    (ranking relativo) y mide la entropía de la distribución de patrones.
    Cercano a 1.0 = aleatorio. < 0.95 = estructura explotable.
    """
    data = closes[-window:]
    n = len(data)
    if n < m + 2:
        return 1.0

    # Construir patrones ordinales
    from collections import Counter
    patterns = []
    for i in range(n - (m - 1) * delay):
        motif = tuple(data[i + j * delay] for j in range(m))
        ranked = tuple(np.argsort(motif).tolist())
        patterns.append(ranked)

    if not patterns:
        return 1.0

    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs))
    import math
    max_entropy = np.log2(float(math.factorial(m))) if m <= 10 else m * np.log2(m)
    if max_entropy < 1e-12:
        return 1.0
    return float(entropy / max_entropy)


def prng_runs_test_z(closes: np.ndarray, window: int = 60) -> float:
    """
    Z-score del test de rachas (Wald-Wolfowitz).

    Verifica si la secuencia de subidas/bajadas es compatible con
    independencia. Valores negativos = demasiadas pocas rachas (tendencia
    a repetir dirección). Valores positivos = demasiadas rachas (oscilación
    forzada / mean-reversion del generador).
    """
    data = closes[-window:]
    if len(data) < 10:
        return 0.0

    median = np.median(data)
    binary = (data > median).astype(int)

    n1 = int(binary.sum())
    n0 = len(binary) - n1
    n = len(binary)

    if n1 == 0 or n0 == 0:
        return 0.0

    runs = 1 + int(np.sum(np.diff(binary) != 0))
    expected = 1.0 + (2.0 * n1 * n0) / n
    var = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / (n**2 * (n - 1))

    if var <= 0:
        return 0.0

    return float((runs - expected) / np.sqrt(var))


def prng_transition_entropy(closes: np.ndarray, window: int = 60) -> float:
    """
    Entropía de la matriz de transición de Markov orden 1.

    Cuenta transiciones UP→UP, UP→DOWN, DOWN→UP, DOWN→DOWN y mide
    la entropía. Valores bajos = transiciones predecibles (el PRNG
    favorece ciertos patrones como reversión tras racha).
    Normalizado a [0, 1]: 1.0 = equiprobable.
    """
    data = closes[-window:]
    if len(data) < 5:
        return 1.0

    dirs = np.sign(np.diff(data))
    dirs = dirs[dirs != 0]  # eliminar dojis

    if len(dirs) < 3:
        return 1.0

    transitions: Dict[tuple, int] = {}
    for i in range(len(dirs) - 1):
        key = (int(dirs[i]), int(dirs[i + 1]))
        transitions[key] = transitions.get(key, 0) + 1

    total = sum(transitions.values())
    if total < 1:
        return 1.0
    probs = np.array([v / total for v in transitions.values()])
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / 2.0)  # max = 2 bits para 4 estados


def prng_hurst_exponent(closes: np.ndarray, window: int = 100) -> float:
    """
    Exponente de Hurst via DFA simplificado.

    H = 0.5: random walk. H > 0.5: persistencia (tendencia).
    H < 0.5: anti-persistencia (mean-reversion).
    Si el PRNG no calibra bien el Hurst, deja sesgo sistemático explotable.
    """
    data = closes[-window:]
    n = len(data)
    if n < 20:
        return 0.5

    scales = [s for s in [4, 8, 16, 32] if s < n // 2]
    if len(scales) < 2:
        return 0.5

    fluctuations = []
    for scale in scales:
        n_segments = n // scale
        rms_list = []
        for i in range(n_segments):
            segment = data[i * scale:(i + 1) * scale].astype(float)
            x = np.arange(scale, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_list.append(rms)
        if rms_list:
            fluctuations.append(np.mean(rms_list))

    if len(fluctuations) < 2 or any(f < 1e-12 for f in fluctuations):
        return 0.5

    log_scales = np.log(np.array(scales[:len(fluctuations)], dtype=float))
    log_fluct = np.log(np.array(fluctuations, dtype=float))
    coeffs = np.polyfit(log_scales, log_fluct, 1)
    h = float(coeffs[0])

    # Clamp a rango razonable [0, 1.5]
    return max(0.0, min(1.5, h))


def prng_turning_point_ratio(closes: np.ndarray, window: int = 60) -> float:
    """
    Proporción de puntos de giro (picos y valles locales).

    Bajo independencia, el ratio esperado es 2/3 ≈ 0.667.
    < 0.667: demasiada inercia (tendencias largas).
    > 0.667: demasiada oscilación (reversiones forzadas por el PRNG).
    """
    data = closes[-window:]
    n = len(data)
    if n < 3:
        return 2.0 / 3.0

    turning_points = 0
    for i in range(1, n - 1):
        if (data[i] > data[i - 1] and data[i] > data[i + 1]) or \
           (data[i] < data[i - 1] and data[i] < data[i + 1]):
            turning_points += 1

    return float(turning_points / (n - 2))


def prng_autocorr_lag(closes: np.ndarray, lag: int, window: int = 60) -> float:
    """
    Autocorrelación de retornos en un lag específico.

    Mersenne Twister y otros PRNG pueden tener correlaciones en
    lags no adyacentes (documentado por Harase para MT19937).
    """
    data = closes[-window:]
    if len(data) < lag + 3:
        return 0.0

    returns = np.diff(data)
    mean_r = returns.mean()
    demeaned = returns - mean_r
    var = float(np.dot(demeaned, demeaned))

    if var < 1e-12:
        return 0.0

    return float(np.dot(demeaned[:-lag], demeaned[lag:]) / var)


def compute_prng_features(df: pd.DataFrame, window: int = 60) -> Dict[str, float]:
    """
    Calcula todas las features PRNG de una sola vez.

    Args:
        df: DataFrame con columna 'close'.
        window: Ventana de cálculo (default 60 velas = 1 hora en M1).

    Returns:
        Dict con las 9 features PRNG.
    """
    closes = df["close"].astype(float).values

    return {
        "prng_last_digit_entropy":   prng_last_digit_entropy(closes, window=window),
        "prng_last_digit_mode_freq": prng_last_digit_mode_freq(closes, window=window),
        "prng_permutation_entropy":  prng_permutation_entropy(closes, window=window),
        "prng_runs_test_z":          prng_runs_test_z(closes, window=window),
        "prng_transition_entropy":   prng_transition_entropy(closes, window=window),
        "prng_hurst_exponent":       prng_hurst_exponent(closes, window=min(100, len(closes))),
        "prng_turning_point_ratio":  prng_turning_point_ratio(closes, window=window),
        "prng_autocorr_lag2":        prng_autocorr_lag(closes, lag=2, window=window),
        "prng_autocorr_lag5":        prng_autocorr_lag(closes, lag=5, window=window),
    }
