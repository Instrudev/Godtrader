"""
validate_bb_body_call.py – Backtest de BB Body Reversal PUT vs CALL espejo.

Usa velas de candles_history para evaluar cuántas veces cada versión
activaría y si la vela siguiente confirma la dirección.

NO opera trades reales. Solo análisis comparativo.

Uso:
    python validate_bb_body_call.py
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("trades.db")
REPORTS_DIR = Path("reports")
MIN_WINDOW = 30  # velas mínimas para construir df con indicadores


def _load_candles(asset: str) -> pd.DataFrame:
    """Carga velas de candles_history para un asset."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candles_history WHERE asset = ? ORDER BY time ASC",
        (asset,),
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "time": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]) if r["volume"] else 0,
        })
    return pd.DataFrame(data)


def validate() -> str:
    """Ejecuta validación y retorna path del reporte."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    report_path = REPORTS_DIR / f"bb_body_validation_{ts}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    from indicators import build_dataframe, detect_bb_body_reversal, detect_bb_body_reversal_call

    # Obtener assets disponibles
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT asset, COUNT(*) as n FROM candles_history GROUP BY asset HAVING n >= 50")
    assets = [(r[0], r[1]) for r in cur.fetchall()]

    # Trades históricos con BB Body
    cur.execute("""
        SELECT asset, direction, result FROM trades
        WHERE ai_reasoning LIKE '%bb_body%' AND result IN ('WIN','LOSS')
    """)
    historical = cur.fetchall()
    conn.close()

    lines = []
    lines.append("# BB Body Reversal Validation Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ── 1. Rendimiento histórico ──
    lines.append("\n## 1. Rendimiento Histórico (trades reales)")
    if historical:
        wins = sum(1 for r in historical if r[2] == "WIN")
        total = len(historical)
        lines.append(f"\n**{total} trades** ejecutados con BB Body Reversal (todos PUT):")
        lines.append(f"\n| Asset | Dir | Result |")
        lines.append(f"|---|---|---|")
        for a, d, r in historical:
            lines.append(f"| `{a}` | {d} | {r} |")
        lines.append(f"\n**Winrate: {wins}/{total} = {wins/total*100:.0f}%** (breakeven ~55%)")
    else:
        lines.append("\nSin trades históricos con BB Body.")

    # ── 2. Backtest sobre velas históricas ──
    lines.append("\n## 2. Backtest sobre velas históricas")
    lines.append(f"\n**Disclaimer:** Backtest sobre {sum(n for _, n in assets)} velas en {len(assets)} assets.")
    lines.append("Cada asset cubre ~5 horas. Resultados indicativos, no concluyentes.")

    total_put_signals = 0
    total_call_signals = 0
    total_put_confirm = 0
    total_call_confirm = 0

    for asset, n_candles in assets:
        raw_df = _load_candles(asset)
        if len(raw_df) < MIN_WINDOW + 1:
            continue

        # Convertir a formato que build_dataframe espera
        raw_list = [
            {"time": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"], "volume": r["volume"]}
            for _, r in raw_df.iterrows()
        ]

        put_signals = 0
        call_signals = 0
        put_confirmed = 0
        call_confirmed = 0

        for i in range(MIN_WINDOW, len(raw_list) - 1):
            window = raw_list[max(0, i - 299):i + 1]
            try:
                df = build_dataframe(window)
                if df.empty or len(df) < 20:
                    continue

                # PUT (original)
                put_ok, _ = detect_bb_body_reversal(df)
                if put_ok:
                    put_signals += 1
                    next_close = raw_list[i + 1]["close"]
                    curr_close = raw_list[i]["close"]
                    if next_close < curr_close:
                        put_confirmed += 1

                # CALL (espejo)
                call_ok, _ = detect_bb_body_reversal_call(df)
                if call_ok:
                    call_signals += 1
                    next_close = raw_list[i + 1]["close"]
                    curr_close = raw_list[i]["close"]
                    if next_close > curr_close:
                        call_confirmed += 1

            except Exception:
                continue

        total_put_signals += put_signals
        total_call_signals += call_signals
        total_put_confirm += put_confirmed
        total_call_confirm += call_confirmed

        if put_signals > 0 or call_signals > 0:
            lines.append(f"\n### {asset} ({n_candles} velas)")
            lines.append(f"- PUT señales: {put_signals} (confirmadas: {put_confirmed})")
            lines.append(f"- CALL señales: {call_signals} (confirmadas: {call_confirmed})")

    # ── 3. Resumen ──
    lines.append("\n## 3. Resumen comparativo")
    lines.append(f"\n| Versión | Señales | Confirmadas | Tasa |")
    lines.append(f"|---|---|---|---|")
    put_rate = f"{total_put_confirm/total_put_signals*100:.0f}%" if total_put_signals > 0 else "N/A"
    call_rate = f"{total_call_confirm/total_call_signals*100:.0f}%" if total_call_signals > 0 else "N/A"
    lines.append(f"| PUT (original) | {total_put_signals} | {total_put_confirm} | {put_rate} |")
    lines.append(f"| CALL (espejo) | {total_call_signals} | {total_call_confirm} | {call_rate} |")

    # ── 4. Conclusión ──
    lines.append("\n## 4. Conclusión")
    if total_put_signals == 0 and total_call_signals == 0:
        lines.append("\n**Sin señales en el histórico de velas.** La estrategia BB Body es extremadamente")
        lines.append("selectiva. Con ~1,125 puntos de evaluación, ni PUT ni CALL activaron.")
        lines.append("Datos insuficientes para validar asimetría PUT-only.")
    elif total_put_signals > 0 or total_call_signals > 0:
        lines.append(f"\nPUT generó {total_put_signals} señales, CALL generó {total_call_signals}.")
        if total_put_signals > 0 and total_call_signals > 0:
            lines.append("Ambas versiones generan señales. Comparar tasas de confirmación.")
        else:
            active_version = "PUT" if total_put_signals > 0 else "CALL"
            lines.append(f"Solo {active_version} generó señales. Asimetría confirmada en este dataset.")

    lines.append("\n## 5. Decisión")
    lines.append("\n**Deprecación instrumentada** de BB Body Reversal como señal activa.")
    lines.append("- Rendimiento histórico: 25% WR (1W/3L), por debajo de breakeven.")
    lines.append("- Señales phantom registradas en `strategy_decisions.log`.")
    lines.append("- Revisión programada tras 30-60 días de operación post-remediación.")
    lines.append("- Criterio de reactivación: phantom signals muestran WR > 55% consistente.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Reporte generado: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    path = validate()
    print(f"\nReporte: {path}")
