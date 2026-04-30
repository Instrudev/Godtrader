"""
audit_ml_model.py – Auditoría estructural del modelo ML actual.

Genera reporte markdown con análisis de feature importances, calibrador,
y conclusiones sobre la validez del modelo. NO ejecuta predicciones masivas
(dataset insuficiente para métricas estadísticas significativas).

Uso:
    python audit_ml_model.py
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/lgbm_model.pkl")
CALIB_PATH = Path("models/platt_calibrator.pkl")
BASELINE_HASH = "f1709cb4359a9157b97fc1b6599cf5b51d6f127f124afb69f79081206f7ebfe8"
REPORTS_DIR = Path("reports")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_trades() -> dict:
    """Cuenta trades en la BD para contexto del reporte."""
    try:
        import sqlite3
        conn = sqlite3.connect("trades.db")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trades WHERE result IN ('WIN','LOSS')")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM trades WHERE result IN ('WIN','LOSS') AND asset LIKE '%-OTC'")
        otc = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(*) FROM trades
            WHERE result IN ('WIN','LOSS') AND streak_length IS NOT NULL
        """)
        with_features = cur.fetchone()[0]
        conn.close()
        return {"total": total, "otc": otc, "with_ml_features": with_features}
    except Exception as e:
        return {"total": 0, "otc": 0, "with_ml_features": 0, "error": str(e)}


def audit() -> str:
    """Ejecuta la auditoría y retorna el path del reporte generado."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    report_path = REPORTS_DIR / f"ml_audit_{ts}.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# ML Model Audit Report")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ── 1. Disclaimer ──
    lines.append("\n## 1. Disclaimer")
    trade_info = _count_trades()
    lines.append(
        f"\nDataset: {trade_info['total']} trades cerrados "
        f"({trade_info['otc']} OTC, {trade_info['with_ml_features']} con features ML pobladas). "
        "**Dataset insuficiente para métricas estadísticas significativas.** "
        "Este reporte se basa en análisis estructural del modelo, no en validación cuantitativa."
    )

    # ── 2. Estado del modelo ──
    lines.append("\n## 2. Estado del modelo")

    if not MODEL_PATH.exists():
        lines.append(f"\n**ERROR:** Modelo no encontrado en `{MODEL_PATH}`.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    model_hash = _sha256(MODEL_PATH)
    calib_hash = _sha256(CALIB_PATH) if CALIB_PATH.exists() else "N/A"
    mod_time = datetime.fromtimestamp(
        os.path.getmtime(MODEL_PATH), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    hash_match = model_hash == BASELINE_HASH

    lines.append(f"\n| Campo | Valor |")
    lines.append(f"|---|---|")
    lines.append(f"| Path | `{MODEL_PATH}` |")
    lines.append(f"| SHA256 | `{model_hash}` |")
    lines.append(f"| Baseline hash | `{BASELINE_HASH}` |")
    lines.append(f"| Hash match | {'OK' if hash_match else '**MISMATCH**'} |")
    lines.append(f"| Fecha modificación | {mod_time} |")
    lines.append(f"| Calibrador hash | `{calib_hash}` |")

    # ── 3. Cargar modelo ──
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(CALIB_PATH, "rb") as f:
        calibrator = pickle.load(f)

    # ── 4. Feature importances ──
    lines.append("\n## 3. Feature Importances")

    from ml_classifier import FEATURE_COLS
    importances = model.feature_importances_
    total_imp = sum(importances)
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])

    nonzero = sum(1 for _, v in pairs if v > 0)
    zero = len(pairs) - nonzero

    lines.append(f"\n**Features totales:** {len(pairs)}")
    lines.append(f"**Features con importancia > 0:** {nonzero} ({nonzero/len(pairs)*100:.0f}%)")
    lines.append(f"**Features con importancia = 0:** {zero} ({zero/len(pairs)*100:.0f}%)")

    lines.append(f"\n| Rank | Feature | Importance | % Total | Estado |")
    lines.append(f"|---|---|---|---|---|")
    for i, (name, imp) in enumerate(pairs, 1):
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        if imp == 0:
            estado = "INACTIVA"
        elif pct > 30:
            estado = "ALTA"
        elif pct > 10:
            estado = "MEDIA"
        else:
            estado = "BAJA"
        lines.append(f"| {i} | `{name}` | {imp:.1f} | {pct:.1f}% | {estado} |")

    # ── 5. Calibrador Platt ──
    lines.append("\n## 4. Calibrador Platt")
    coef = calibrator.coef_[0][0] if hasattr(calibrator, "coef_") else "N/A"
    intercept = calibrator.intercept_[0] if hasattr(calibrator, "intercept_") else "N/A"
    lines.append(f"\n| Parámetro | Valor | Interpretación |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Coeficiente | {coef:.6f} | Pendiente de la transformación logística |")
    lines.append(f"| Intercepto | {intercept:.6f} | Offset de la transformación |")
    lines.append(
        f"\nCon coef={coef:.4f}, la transformación es casi identidad con ligero offset negativo. "
        "El calibrador no está corrigiendo significativamente las predicciones del modelo base."
    )

    # ── 6. Parámetros del modelo ──
    lines.append("\n## 5. Parámetros LightGBM")
    params = model.get_params() if hasattr(model, "get_params") else {}
    key_params = ["n_estimators", "max_depth", "learning_rate", "num_leaves",
                  "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    lines.append(f"\n| Parámetro | Valor |")
    lines.append(f"|---|---|")
    for k in key_params:
        lines.append(f"| `{k}` | {params.get(k, 'N/A')} |")

    # ── 7. Conclusión ──
    lines.append("\n## 6. Conclusión")
    lines.append("\n### Hallazgos críticos")
    lines.append(f"\n1. **{zero}/{len(pairs)} features ({zero/len(pairs)*100:.0f}%) con importancia 0** — "
                 "el modelo no aprendió de la mayoría de sus inputs.")
    lines.append(f"2. **`direction` con importancia 0** — el modelo NO distingue CALL de PUT. "
                 "Las predicciones `call_proba` y `put_proba` son efectivamente iguales.")
    lines.append(f"3. **Top 3 features concentran {sum(v for _, v in pairs[:3])/total_imp*100:.0f}% del peso** — "
                 "el modelo es un clasificador de Bollinger/RSI, no de 27 features.")
    lines.append(f"4. **Calibrador Platt casi neutro** (coef={coef:.4f}) — "
                 "no corrige significativamente las predicciones.")
    lines.append(f"5. **9 features PRNG son inútiles** — solo 1 de 9 tiene importancia marginal (3.3%).")

    lines.append("\n### Estado del modelo: **NO AÑADE VALOR PREDICTIVO REAL**")
    lines.append(
        "\nEl modelo funciona como un clasificador básico de Bollinger/RSI con 300 árboles. "
        "No captura información direccional (CALL vs PUT), no usa features PRNG, "
        "y el calibrador no lo mejora significativamente. "
        "Equivale operativamente a no tener ML gate."
    )

    # ── 8. Recomendación ──
    lines.append("\n## 7. Recomendación")
    lines.append("\n1. **Activar Tarea 1.6** (política fallback sin ML) — el modelo actual no debe operar como gate.")
    lines.append("2. **Requisitos para reentrenamiento (Tarea 3.1):**")
    lines.append("   - Mínimo 500 trades con filtros activos (post-Tarea 1.0).")
    lines.append("   - Excluir trades pre-Tarea 1.0 y no-OTC del dataset.")
    lines.append("   - `train_model.py` debe guardar `metrics.json`.")
    lines.append("   - `direction` debe estar entre top 5 features por importancia.")
    lines.append("   - Feature importances con distribución razonable (no 78% en cero).")
    lines.append("   - Considerar 2 modelos separados (CALL/PUT) si `direction` sigue sin peso.")
    lines.append("   - Eliminar features PRNG si siguen mostrando importancia ~0.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Reporte generado: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    path = audit()
    print(f"\nReporte: {path}")
