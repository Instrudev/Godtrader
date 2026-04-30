"""
ai_brain.py – Motor de Decisión IA con Ollama (LLM Local)

DEPRECATED: This module is deprecated as of remediation/v1.

ai_brain.py is only invoked by trader.py (now deprecated) as an LLM
fallback when the ML classifier is unavailable. Using an LLM as a
trading decision gate is architecturally questionable and not
validated under the current remediation plan.

See CHANGELOG_REMEDIATION.md for details.
"""

import warnings
from iqservice import ALLOW_DEPRECATED_TRADERS

if not ALLOW_DEPRECATED_TRADERS:
    raise ImportError(
        f"{__name__} is deprecated and disabled during remediation. "
        "Use asset_scanner.py instead. "
        "If you have a valid reason to use this module, set "
        "ALLOW_DEPRECATED_TRADERS=True in iqservice.py with documented justification."
    )

warnings.warn(
    f"{__name__} is deprecated. Use asset_scanner.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import logging
import os
import re
from typing import Dict, List, Optional
import ollama
import pandas as pd
from dotenv import load_dotenv

from indicators import detect_patterns, find_support_resistance, calculate_cycle_stats, calculate_adherence_index, detect_vsa_anomaly

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Configuración de Ollama ──────────────────────────────────────────────────

OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Parámetros optimizados para Apple M5 (Metal GPU via Ollama)
# num_gpu=999 → usar todas las capas disponibles en GPU unificada
# temperature=0.05 → máximo determinismo en decisiones de trading
# num_predict=120 → la respuesta JSON es corta; no desperdiciar ciclos
_OLLAMA_OPTIONS = {
    "temperature":   0.05,
    "num_predict":   120,
    "num_ctx":       2048,
    "num_gpu":       999,   # Metal (Apple Silicon): todas las capas en GPU
    "num_thread":    8,     # M5 tiene 10 cores; dejar 2 para el SO
    "repeat_penalty": 1.1,
    "stop":          ["\n```\n", "```json"],
}

# ─── System Prompt – Analista Institucional ───────────────────────────────────

_SYSTEM_PROMPT = """Eres un motor de decisión de trading OTC. Devuelves ÚNICAMENTE un objeto JSON de una sola línea sin texto adicional, sin markdown, sin explicaciones.

FORMAT: {"op":"CALL","pr":87,"ex":2,"an":"descripcion breve"}
CAMPOS: op=CALL|PUT|WAIT, pr=0-100, ex=1-5, an=texto libre max 60 chars.

REGLAS (primera que aplica, gana):
R0: RSI>75 → op=WAIT. RSI<25 → op=WAIT.
R1: consec_bull>5 → no CALL → WAIT. consec_bear>5 → no PUT → WAIT.
R2: EMA20_Gap_pips>12 → op=WAIT.
R3: sentimiento CAZADOR o ALTA_FRICCION → pr>=95 o op=WAIT.
R4: tendencia BULL → busca CALL. tendencia BEAR → busca PUT.
R5: Shooting_Star o Bear_Engulf → PUT. Hammer o Bull_Engulf → CALL. Patron contradice op → WAIT.
R6: RSI overbought → no CALL. RSI oversold → no PUT.
R7: historial perdida <1pip → misma direccion, cambia ex. >5pips → invierte o WAIT.
R8: vol_rel alto + vela grande → ex=1 o 2. mercado lento/mechas → ex=3 a 5.
R9: pr<78 → op=WAIT.
R10: RSI extremo y propones entrada → pr -= 30."""


# ─── Función Principal ────────────────────────────────────────────────────────

def get_ai_decision(df: pd.DataFrame, asset: str) -> Optional[Dict]:
    """
    Envía el snapshot de 3 bloques a Ollama y retorna la decisión de trading.

    El modelo phi3 corre localmente en el chip M5 via Metal (GPU unificada),
    eliminando latencia de red y costos de API.

    Args:
        df:    DataFrame con indicadores calculados (build_dataframe).
        asset: Nombre del activo (e.g., "EURUSD").

    Returns:
        dict con claves {op, pr, ex, an} o None si hay error.
    """
    if df.empty or len(df) < 205:
        logger.warning(f"Datos insuficientes: {len(df)} velas (mínimo 205)")
        return None

    snapshot = _build_3block_snapshot(df, asset)
    logger.debug(f"Snapshot enviado a Ollama:\n{snapshot}")

    raw_response = ""
    try:
        client   = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": snapshot},
            ],
            options=_OLLAMA_OPTIONS,
        )
        raw_response = response["message"]["content"].strip()
        logger.debug(f"Respuesta cruda Ollama: {raw_response!r}")

        decision = _extract_json(raw_response)
        _validate_decision(decision)

        # Normalizar tipos
        decision["pr"] = int(decision["pr"])
        decision["ex"] = int(decision["ex"])
        decision["op"] = str(decision["op"]).upper()

        return decision

    except ollama.ResponseError as e:
        logger.error(f"Ollama ResponseError: {e.error}")
        return None
    except ConnectionError:
        logger.error(
            f"No se puede conectar a Ollama en {OLLAMA_HOST}. "
            f"¿Está corriendo 'ollama serve'?"
        )
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON inválido de Ollama: {e} | Respuesta: {raw_response!r}")
        return None
    except ValueError as e:
        logger.error(f"Schema inválido: {e}")
        return None


# ─── Extracción y Validación del JSON ────────────────────────────────────────

def _extract_json(text: str) -> Dict:
    """
    Extrae el primer objeto JSON válido del texto de respuesta del LLM.
    Maneja casos donde el modelo incluye texto extra antes/después del JSON.
    """
    # Intentar parseo directo primero (respuesta limpia)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Buscar el primer bloque JSON con regex
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def _validate_decision(d: Dict) -> None:
    """Valida que la decisión cumpla el schema esperado. Lanza ValueError si no.
    El campo 'an' es opcional y se rellena con string vacío si falta.
    """
    required = {"op", "pr", "ex"}
    missing  = required - d.keys()
    if missing:
        raise ValueError(f"Campos faltantes: {missing}")
    # 'an' es opcional — rellenar con vacío si el modelo lo omitió
    if "an" not in d:
        d["an"] = ""
    if d["op"] not in ("CALL", "PUT", "WAIT"):
        raise ValueError(f"op inválido: {d['op']!r}")
    if not (0 <= int(d["pr"]) <= 100):
        raise ValueError(f"pr fuera de rango: {d['pr']}")
    # Clamp ex al rango válido en lugar de rechazar
    d["ex"] = max(1, min(5, int(d["ex"])))


# ─── Constructor del Snapshot de 3 Bloques ───────────────────────────────────

def _build_3block_snapshot(df: pd.DataFrame, asset: str) -> str:
    """
    Construye el mensaje de 3 bloques minificados optimizado para:
    - Mínimo uso de tokens (reducir costo computacional en M5)
    - Máxima información relevante para el análisis institucional

    Bloque MACRO:  tendencia macro, volatilidad, S/R histórico
    Bloque TECH:   RSI, Bollinger, Volumen Relativo
    Bloque MICRO:  OHLC de las últimas 10 velas + patrones detectados
    Bloque ERROR:  Resumen de los últimos 3 trades perdedores (Self-Learning)
    """
    last = df.iloc[-1]

    price    = float(last["close"])
    ema20    = float(last["ema20"])
    ema200   = float(last["ema200"])
    rsi      = float(last["rsi"])
    bb_upper = float(last["bb_upper"])
    bb_mid   = float(last["bb_mid"])
    bb_lower = float(last["bb_lower"])
    bb_width = float(last["bb_width"])
    vol_rel  = float(last["vol_rel"])

    trend      = "BULL" if price > ema200 else "BEAR"
    ema_dist   = round(abs(price - ema200) / ema200 * 100, 4)
    timestamp  = last["time"].strftime("%Y-%m-%d %H:%M UTC")

    # ── Soporte / Resistencia ─────────────────────────────────────────────────
    sr          = find_support_resistance(df)
    resistances = [round(r, 5) for r in sr["resistances"]]
    supports    = [round(s, 5) for s in sr["supports"]]

    # ── Posición en Bollinger ─────────────────────────────────────────────────
    if price >= bb_upper:
        bb_pos = "AT_UPPER"
    elif price >= bb_mid:
        bb_pos = "ABOVE_MID"
    elif price > bb_lower:
        bb_pos = "BELOW_MID"
    else:
        bb_pos = "AT_LOWER"

    # ── Zona RSI ──────────────────────────────────────────────────────────────
    rsi_zone = (
        "OVERSOLD"   if rsi < 35 else
        "OVERBOUGHT" if rsi > 65 else
        "NEUTRAL"
    )

    # ── Patrones de vela (últimas 10 velas) ───────────────────────────────────
    patterns     = detect_patterns(df, lookback=10)
    pattern_strs = [f"{p['pattern']}@{p['time']}({p['bias']})" for p in patterns]
    pattern_val  = pattern_strs if pattern_strs else ["NONE"]

    # ── Últimas 10 velas OHLC (confirmadas, excluye vela en formación) ─────────
    micro_candles = []
    for _, row in df.iloc[-11:-1].iterrows():
        arrow = "▲" if row["close"] >= row["open"] else "▼"
        micro_candles.append({
            "t":  row["time"].strftime("%H:%M"),
            "d":  arrow,
            "o":  round(float(row["open"]),  5),
            "h":  round(float(row["high"]),  5),
            "l":  round(float(row["low"]),   5),
            "c":  round(float(row["close"]), 5),
            "vr": round(float(row["vol_rel"]), 2),
        })

    # ── Ensamblado de los 3 bloques ───────────────────────────────────────────
    macro_block = json.dumps({
        "asset":   asset,
        "ts":      timestamp,
        "trend":   trend,
        "price":   round(price,  5),
        "ema20":   round(ema20,  5),
        "ema200":  round(ema200, 5),
        "dist%":   ema_dist,
        "bb_w%":   round(bb_width, 3),
        "res":     resistances,
        "sup":     supports,
    }, separators=(",", ":"))

    tech_block = json.dumps({
        "rsi":     round(rsi, 2),
        "rsi_z":   rsi_zone,
        "bb_pos":  bb_pos,
        "bb_u":    round(bb_upper, 5),
        "bb_m":    round(bb_mid,   5),
        "bb_l":    round(bb_lower, 5),
        "vol_rel": round(vol_rel,  2),
        "patterns": pattern_val,
    }, separators=(",", ":"))

    # ── Módulo Anti-Fatiga y Distancia a EMA20 (Cero Tolerancia) ──────────────
    consec_bull = 0
    consec_bear = 0
    for _, row_d in df.iloc[::-1].iterrows():
        o = float(row_d["open"])
        c = float(row_d["close"])
        if c > o:
            if consec_bear > 0: break
            consec_bull += 1
        elif c < o:
            if consec_bull > 0: break
            consec_bear += 1
        else:
            break
            
    pips_to_ema20 = abs(price - ema20) * (100 if price > 10 else 10000)
    
    micro_block = json.dumps({
        "candles": micro_candles,
        "consec_bull": consec_bull,
        "consec_bear": consec_bear,
        "EMA20_Gap_pips": round(pips_to_ema20, 2)
    }, separators=(",", ":"))

    # ── Módulo de Memoria de Ciclo ────────────────────────────────────────────
    cycle = calculate_cycle_stats(df)
    cycle_block = json.dumps({
        "rsi_respect_rate": f"{cycle['rsi_respect_rate']}%",
        "avg_breakout_dist": f"{cycle['bb_breakout_avg']} px",
        "market_state": cycle["market_state"]
    }, separators=(',', ':'))

    # ── Módulo de Memoria de Errores (Self-Learning) ──────────────────────────
    errors_str = "Ninguno"
    friccion = False
    try:
        import os
        if os.path.exists("trades_history.json"):
            with open("trades_history.json", "r") as f:
                history = json.load(f)
                if history:
                    recent = history[-3:]
                    lines = []
                    friccion_count = 0
                    for e in recent:
                        pips = e.get('pips_difference', 100)
                        if pips < 0.8: friccion_count += 1
                        lines.append(f"{e['failed_op']}_fallo_por_{pips:.1f}_pips")
                    errors_str = ",".join(lines)
                    friccion = friccion_count >= 2
    except:
        pass
    error_block = json.dumps({"recents": errors_str}, separators=(',', ':'))

    # ── Intencionalidad del Broker (Market Making) ────────────────────────────
    adherencia = calculate_adherence_index(df)
    vsa_abnormal = detect_vsa_anomaly(df)
    broker_block = json.dumps({
        "Adherencia": adherencia,
        "VSA_Anomaly": vsa_abnormal,
        "Friction_Mode": "ALTA_FRICCIÓN" if friccion else "NORMAL"
    }, separators=(',', ':'))

    snapshot = (
        f"[CICLO]{cycle_block}\n"
        f"[MACRO]{macro_block}\n"
        f"[TECH]{tech_block}\n"
        f"[MICRO]{micro_block}\n"
        f"[HISTORIAL DE ERRORES]{error_block}\n"
        f"[SENTIMIENTO_DEL_ALGORITMO_BROKER]{broker_block}"
    )

    return snapshot
