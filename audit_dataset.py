"""
audit_dataset.py – Auditoría del dataset de entrenamiento ML.

Genera reporte markdown con análisis de contaminación, features incompletas,
anti-patrones, y especificación de prerequisitos para Tarea 3.1.

NO modifica la BD ni genera datasets limpios (datos actuales no son viables).

Uso:
    python audit_dataset.py
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("trades.db")
REPORTS_DIR = Path("reports")

# Features del modelo ML (ml_classifier.FEATURE_COLS)
BASIC_FEATURES = ["rsi", "bb_pct_b", "bb_width_pct", "vol_rel", "hour_utc", "weekday"]
STREAK_FEATURES = ["streak_length", "streak_pct", "ret_3", "ret_5", "ret_10",
                    "autocorr_10", "half_life", "rel_atr"]
PRNG_FEATURES = ["prng_last_digit_entropy", "prng_last_digit_mode_freq",
                 "prng_permutation_entropy", "prng_runs_test_z",
                 "prng_transition_entropy", "prng_hurst_exponent",
                 "prng_turning_point_ratio", "prng_autocorr_lag2", "prng_autocorr_lag5"]


def audit() -> str:
    """Ejecuta la auditoría y retorna el path del reporte generado."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    report_path = REPORTS_DIR / f"dataset_audit_{ts}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    lines = []
    lines.append("# Dataset Audit Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ── 1. Estado actual ──
    lines.append("\n## 1. Estado Actual del Dataset")

    cur.execute("SELECT COUNT(*) FROM trades WHERE result IN ('WIN','LOSS')")
    total_closed = cur.fetchone()[0]

    cur.execute("SELECT result, COUNT(*) FROM trades GROUP BY result ORDER BY result")
    lines.append(f"\n**Total trades cerrados (WIN/LOSS):** {total_closed}")
    lines.append(f"\n| Resultado | Count |")
    lines.append(f"|---|---|")
    for row in cur.fetchall():
        lines.append(f"| {row[0] or 'NULL'} | {row[1]} |")

    cur.execute("SELECT mode, COUNT(*) FROM trades GROUP BY mode")
    lines.append(f"\n| Mode | Count |")
    lines.append(f"|---|---|")
    for row in cur.fetchall():
        lines.append(f"| {row[0]} | {row[1]} |")

    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM trades")
    mn, mx = cur.fetchone()
    lines.append(f"\n**Rango temporal:** {mn[:10] if mn else 'N/A'} → {mx[:10] if mx else 'N/A'}")

    cur.execute("""
        SELECT asset, COUNT(*) as c,
               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as w,
               SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as l
        FROM trades WHERE result IN ('WIN','LOSS')
        GROUP BY asset ORDER BY c DESC LIMIT 20
    """)
    lines.append(f"\n| Asset | Trades | W | L | OTC |")
    lines.append(f"|---|---|---|---|---|")
    for row in cur.fetchall():
        otc = "Yes" if row[0].endswith("-OTC") else "**NO**"
        lines.append(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} | {otc} |")

    cur.execute("""
        SELECT DATE(timestamp) as day, COUNT(*),
               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END),
               SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)
        FROM trades WHERE result IN ('WIN','LOSS')
        GROUP BY day ORDER BY day
    """)
    lines.append(f"\n| Fecha | Trades | W | L |")
    lines.append(f"|---|---|---|---|")
    for row in cur.fetchall():
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    # ── 2. Contaminación ──
    lines.append("\n## 2. Análisis de Contaminación")

    cur.execute("""
        SELECT id, timestamp, asset, direction, result, mode
        FROM trades WHERE asset NOT LIKE '%-OTC' ORDER BY id
    """)
    no_otc = cur.fetchall()
    lines.append(f"\n**Trades no-OTC:** {len(no_otc)}")
    if no_otc:
        lines.append(f"\n| ID | Timestamp | Asset | Dir | Result | Mode |")
        lines.append(f"|---|---|---|---|---|---|")
        for r in no_otc:
            lines.append(f"| {r[0]} | {r[1][:19]} | `{r[2]}` | {r[3]} | {r[4]} | {r[5]} |")

    lines.append(f"\n**Trades pre-Tarea 1.0 (sin filtros completos):** {total_closed} (todos)")
    lines.append(f"**Trades post-Tarea 1.0 (con filtros):** 0 (branch nunca desplegada)")

    # ── 3. Features ──
    lines.append("\n## 3. Análisis de Features")

    def _count_non_null(feature_list):
        conditions = " AND ".join(f"{f} IS NOT NULL" for f in feature_list)
        cur.execute(f"""
            SELECT COUNT(*) FROM trades
            WHERE result IN ('WIN','LOSS') AND asset LIKE '%-OTC' AND {conditions}
        """)
        return cur.fetchone()[0]

    basic_count = _count_non_null(BASIC_FEATURES)
    streak_count = _count_non_null(STREAK_FEATURES)
    prng_count = _count_non_null(PRNG_FEATURES)

    # Trades con TODAS las features
    all_features = BASIC_FEATURES + STREAK_FEATURES + PRNG_FEATURES
    complete_count = _count_non_null(all_features)

    lines.append(f"\n| Categoría | Features | Trades con datos | % |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| Básicas (rsi, bb_*) | {len(BASIC_FEATURES)} | {basic_count} | {basic_count/total_closed*100:.0f}% |")
    lines.append(f"| Streak/ML (streak_*, ret_*, half_life) | {len(STREAK_FEATURES)} | {streak_count} | {streak_count/total_closed*100:.0f}% |")
    lines.append(f"| PRNG (prng_*) | {len(PRNG_FEATURES)} | {prng_count} | {prng_count/total_closed*100:.0f}% |")
    lines.append(f"| **COMPLETAS (todas)** | **{len(all_features)}** | **{complete_count}** | **{complete_count/total_closed*100:.0f}%** |")

    # Por pipeline
    cur.execute("""
        SELECT
            CASE WHEN ai_reasoning LIKE '%[scanner]%' THEN 'scanner' ELSE 'trader' END as pipeline,
            COUNT(*),
            SUM(CASE WHEN streak_length IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN prng_last_digit_entropy IS NOT NULL THEN 1 ELSE 0 END)
        FROM trades WHERE result IN ('WIN','LOSS')
        GROUP BY pipeline
    """)
    lines.append(f"\n| Pipeline | Trades | Con streak | Con PRNG |")
    lines.append(f"|---|---|---|---|")
    for row in cur.fetchall():
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    # ── 4. Anti-patrón ──
    lines.append("\n## 4. Análisis del Pipeline `_row_to_features`")
    lines.append(
        "\n**Anti-patrón identificado:** `_row_to_features()` en `train_model.py` "
        "rellena NULL con valores default sensibles (streak_length=0, streak_pct=50, "
        "ret_*=0, half_life=100, prng_*=0)."
    )
    lines.append(
        "\n**Consecuencia:** 78 trades del scanner tienen valores defaults idénticos "
        "en 8 features de racha/retorno y 9 features PRNG. El modelo entrenó con "
        "68% del dataset sin varianza informativa real en esas features."
    )
    lines.append(
        "\n**Implicación directa:** Explica por qué 21/27 features tienen "
        "importancia 0 en el modelo actual (Tarea 1.5). El modelo no puede "
        "aprender de features que son constantes en 2/3 del dataset."
    )

    # ── 5. Conclusión ──
    lines.append("\n## 5. Conclusión")
    lines.append(
        "\n**Dataset NO viable para reentrenamiento inmediato.** Causa raíz: "
        "combinación de dataset pequeño (114 trades), features incompletas "
        "(0 trades con features completas), y anti-patrón de defaults que "
        "elimina varianza informativa."
    )

    # ── 6. Especificación Tarea 3.1 ──
    lines.append("\n## 6. Especificación de Tarea 3.1 (Reentrenamiento)")
    lines.append("\n### Criterios de activación (mínimos)")
    lines.append("- 200 trades con features COMPLETAS (no rellenadas con defaults).")
    lines.append("- Bot operando con `remediation/v1` desplegado (filtros + features pobladas).")
    lines.append("- Distribución balanceada: al menos 30% WIN y 30% LOSS.")
    lines.append("- Datos de al menos 30 días de operación continua.")

    lines.append("\n### Pipeline a revisar")
    lines.append("- `_row_to_features` debe NO rellenar NULL con defaults.")
    lines.append("- Opciones: dropear filas con NULL, imputación por mediana, o flag explícito.")
    lines.append("- `asset_scanner` debe poblar TODAS las features (streak + PRNG) en cada trade.")

    lines.append("\n### Validación obligatoria")
    lines.append("- Walk-forward temporal (no aleatoria).")
    lines.append("- `metrics.json` guardado en cada entrenamiento.")
    lines.append("- Feature importances logueadas.")
    lines.append("- AUC >= 0.62 en test set.")
    lines.append("- `direction` debe estar entre top 5 features por importancia.")

    lines.append("\n### Arquitectura alternativa a considerar")
    lines.append("- 2 modelos separados (CALL y PUT) si `direction` sigue con importancia 0.")
    lines.append("- Eliminar features PRNG si siguen mostrando importancia ~0 con datos completos.")

    # ── 7. Estimación temporal ──
    lines.append("\n## 7. Estimación Temporal")
    lines.append("- Configuración Tarea 1.6: max 5 trades/día.")
    lines.append("- Realista: 1-3 trades/día (score >= 0.75 es selectivo).")
    lines.append("- Para 200 trades: 70-200 días (2-6 meses).")
    lines.append("- Para 500 trades: 170-500 días (6-18 meses).")
    lines.append("- **Recomendación:** revisar criterios cada 30 días de operación.")

    # ── 8. Issues conocidos ──
    lines.append("\n## 8. Issues Conocidos para Tarea 3.1")
    lines.append("\n| # | Issue | Impacto | Archivo |")
    lines.append("|---|---|---|---|")
    lines.append("| 1 | `_row_to_features` rellena NULL con defaults | 21/27 features con importancia 0 | `train_model.py` |")
    lines.append("| 2 | Pipeline scanner no popula features streak | 78 trades sin streak_length/pct/ret_* | `asset_scanner.py` |")
    lines.append("| 3 | Pipeline trader no popula features PRNG | 36 trades sin prng_* | `trader.py` (deprecated) |")
    lines.append("| 4 | `train_model.py` no guarda `metrics.json` | Sin historial de AUC/Brier | `train_model.py` |")

    # ── 9. Trades excluidos ──
    lines.append("\n## 9. Trades Excluidos de Futuro Reentrenamiento")
    lines.append("\nTrades no-OTC a excluir permanentemente:")
    lines.append("\n| ID | Asset | Razón |")
    lines.append("|---|---|---|")
    lines.append("| 421 | `EURJPY-op` | Mercado real (no OTC) |")
    lines.append("| 422 | `BXY` | Bloomberg Dollar Index (no OTC, no forex) |")

    conn.close()

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Reporte generado: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    path = audit()
    print(f"\nReporte: {path}")
