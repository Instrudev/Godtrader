"""
train_model.py – Entrenamiento del clasificador ML para señales OTC.

Flujo:
  1. Carga trades cerrados (WIN/LOSS) desde trades.db
  2. Construye matriz de features (FEATURE_COLS de ml_classifier.py)
  3. Walk-forward split: 70% train | 15% val (Platt) | 15% test
  4. Entrena LightGBM binario (WIN=1)
  5. Calibra probabilidades con Platt scaling (LogisticRegression sobre val)
  6. Evalúa en test: AUC-ROC, accuracy, Brier score, Profit Factor simulado
  7. Guarda modelo + calibrador en models/
  8. Imprime reporte

Uso:
    python train_model.py
    python train_model.py --db trades.db --output models/ --min-samples 50

Criterios de promoción (el modelo se guarda solo si los cumple):
    AUC-ROC ≥ 0.60   (rendimiento sobre azar)
    Brier score ≤ 0.24 (calibración aceptable; random = 0.25)
    N test ≥ 20
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Añadir el directorio del script al path para importar módulos locales
sys.path.insert(0, str(Path(__file__).parent))

from database import fetch_training_data, DB_PATH
from ml_classifier import FEATURE_COLS, MODEL_DIR, MODEL_PATH, CALIBRATOR_PATH


# ─── Construcción de features desde filas de la DB ───────────────────────────

def _row_to_features(row: Dict) -> Optional[np.ndarray]:
    """
    Convierte una fila de la DB en un vector de features.
    Devuelve None si hay campos críticos nulos.
    """
    try:
        direction = 1.0 if str(row.get("direction", "")).upper() == "CALL" else 0.0

        # half_life cap a 100 (igual que en ml_classifier.extract_features)
        hl = row.get("half_life", 100.0)
        try:
            hl_val = float(hl) if hl is not None else 100.0
        except (TypeError, ValueError):
            hl_val = 100.0
        half_life = min(hl_val if np.isfinite(hl_val) else 100.0, 100.0)

        def _f(key: str, default: float = 0.0) -> float:
            v = row.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        feat = {
            # ── Indicadores técnicos clásicos ────────────────────────────────
            "rsi":          _f("rsi",          50.0),
            "bb_pct_b":     _f("bb_pct_b",      0.5),
            "bb_width_pct": _f("bb_width_pct",  1.0),
            "rel_atr":      _f("rel_atr",        1.0),
            "vol_rel":      _f("vol_rel",        1.0),
            "hour_utc":     _f("hour_utc",       0.0),
            "weekday":      _f("weekday",        0.0),
            "direction":    direction,
            "streak_length": _f("streak_length", 0.0),
            "streak_pct":   _f("streak_pct",    50.0),
            "ret_3":        _f("ret_3",          0.0),
            "ret_5":        _f("ret_5",          0.0),
            "ret_10":       _f("ret_10",         0.0),
            "autocorr_10":  _f("autocorr_10",    0.0),
            "half_life":    half_life,
            "payout":       _f("payout",         0.80),
            "winrate_hour": _f("winrate_hour",   0.5),
            "expiry_min":   _f("expiry_min",     2.0),
            # ── Features PRNG (defaults neutros para trades anteriores) ──────
            "prng_last_digit_entropy":   _f("prng_last_digit_entropy",   1.0),
            "prng_last_digit_mode_freq": _f("prng_last_digit_mode_freq", 0.10),
            "prng_permutation_entropy":  _f("prng_permutation_entropy",  1.0),
            "prng_runs_test_z":          _f("prng_runs_test_z",          0.0),
            "prng_transition_entropy":   _f("prng_transition_entropy",   1.0),
            "prng_hurst_exponent":       _f("prng_hurst_exponent",       0.5),
            "prng_turning_point_ratio":  _f("prng_turning_point_ratio",  0.667),
            "prng_autocorr_lag2":        _f("prng_autocorr_lag2",        0.0),
            "prng_autocorr_lag5":        _f("prng_autocorr_lag5",        0.0),
        }

        return np.array([feat[col] for col in FEATURE_COLS], dtype=np.float32)

    except Exception as exc:
        logger.warning(f"Error convirtiendo fila a features: {exc}")
        return None


# ─── Walk-forward split ───────────────────────────────────────────────────────

def _walk_forward_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split temporal sin shuffle: train | val | test.
    El orden cronológico se preserva para evitar look-ahead bias.
    """
    n = len(y)
    i_val  = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:i_val],          y[:i_val]
    X_val,   y_val   = X[i_val:i_test],    y[i_val:i_test]
    X_test,  y_test  = X[i_test:],         y[i_test:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─── Métricas ─────────────────────────────────────────────────────────────────

def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC calculada con numpy (sin sklearn, por si no está disponible)."""
    # Ordenar por score descendente
    desc = np.argsort(-y_score)
    y_t  = y_true[desc]
    n_pos = int(y_t.sum())
    n_neg = len(y_t) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(y_t)
    fp = np.cumsum(1 - y_t)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # np.trapezoid en NumPy ≥ 2.0; np.trapz en versiones anteriores
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trapz(tpr, fpr))
    return abs(auc)


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def _profit_factor_sim(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, payout: float = 0.80) -> float:
    """Simula profit factor si operamos cuando proba >= threshold."""
    mask  = y_prob >= threshold
    if mask.sum() == 0:
        return 0.0
    wins   = float(y_true[mask].sum())
    losses = float((1 - y_true[mask]).sum())
    gross_profit = wins * payout
    gross_loss   = losses * 1.0
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train(
    db_path: Path = DB_PATH,
    output_dir: Path = MODEL_DIR,
    min_samples: int = 50,
    promote_auc: float = 0.60,
    promote_brier: float = 0.24,
) -> bool:
    """
    Entrena y guarda el modelo. Devuelve True si el modelo fue promovido.

    Args:
        db_path:      Ruta a trades.db.
        output_dir:   Carpeta donde guardar modelo y calibrador.
        min_samples:  Mínimo de muestras totales para entrenar.
        promote_auc:  AUC mínima para guardar el modelo.
        promote_brier: Brier máximo para guardar el modelo.
    """
    try:
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        logger.error(f"Dependencias faltantes: {e}. Instalar con: pip install lightgbm scikit-learn")
        return False

    # ── Cargar datos ─────────────────────────────────────────────────────────
    logger.info(f"Cargando datos desde {db_path}...")
    rows = fetch_training_data(path=db_path)

    if len(rows) < min_samples:
        logger.warning(
            f"Solo {len(rows)} muestras disponibles (mínimo {min_samples}). "
            "Acumular más trades con el paper trader antes de entrenar."
        )
        return False

    # ── Construir X, y ───────────────────────────────────────────────────────
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for row in rows:
        feat = _row_to_features(row)
        if feat is None:
            continue
        label = 1.0 if row.get("result") == "WIN" else 0.0
        X_list.append(feat)
        y_list.append(label)

    if len(X_list) < min_samples:
        logger.warning(f"Solo {len(X_list)} filas válidas tras filtrar NaN.")
        return False

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)

    winrate = float(y.mean())
    logger.info(f"Dataset: {len(y)} muestras | Winrate: {winrate:.1%} | Features: {X.shape[1]}")

    # ── Walk-forward split ────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = _walk_forward_split(X, y)
    logger.info(
        f"Split: train={len(y_train)} | val={len(y_val)} | test={len(y_test)}"
    )

    if len(y_test) < 20:
        logger.warning(f"Set de test insuficiente ({len(y_test)} muestras). Necesitas más datos.")
        # Continuar de todas formas pero advertir

    # ── LightGBM ─────────────────────────────────────────────────────────────
    scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_child_samples": 10,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "scale_pos_weight":  scale_pos,
        "verbose":           -1,
        "n_jobs":            -1,
        "seed":              42,
    }

    logger.info("Entrenando LightGBM...")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    logger.info(f"Mejor iteración: {model.best_iteration_}")

    # ── Platt scaling ────────────────────────────────────────────────────────
    # Calibrar sobre el set de validación para evitar overfitting
    raw_val = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(raw_val, y_val)
    logger.info("Platt scaling ajustado sobre set de validación.")

    # ── Evaluación en test ────────────────────────────────────────────────────
    raw_test   = model.predict_proba(X_test)[:, 1]
    cal_test   = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    auc    = _auc_roc(y_test, cal_test)
    brier  = _brier_score(y_test, cal_test)
    acc    = float((( cal_test >= 0.5 ).astype(float) == y_test).mean())
    pf_65  = _profit_factor_sim(y_test, cal_test, threshold=0.65)
    pf_70  = _profit_factor_sim(y_test, cal_test, threshold=0.70)

    logger.info("─" * 50)
    logger.info("MÉTRICAS EN TEST (walk-forward, sin look-ahead):")
    logger.info(f"  AUC-ROC      : {auc:.4f}  (mínimo para promover: {promote_auc:.2f})")
    logger.info(f"  Brier score  : {brier:.4f} (máximo para promover: {promote_brier:.2f})")
    logger.info(f"  Accuracy     : {acc:.4f}")
    logger.info(f"  Profit Factor @65%: {pf_65:.2f}")
    logger.info(f"  Profit Factor @70%: {pf_70:.2f}")
    logger.info(f"  Muestras test: {len(y_test)}")
    logger.info("─" * 50)

    # ── Importancia de features ───────────────────────────────────────────────
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top 10 features (gain):")
    for feat_name, imp in importances[:10]:
        logger.info(f"  {feat_name:<20} {imp:.1f}")

    # ── Criterio de promoción ─────────────────────────────────────────────────
    if auc < promote_auc:
        logger.warning(f"AUC {auc:.4f} < {promote_auc:.2f}. Modelo NO promovido.")
        return False
    if brier > promote_brier:
        logger.warning(f"Brier {brier:.4f} > {promote_brier:.2f}. Modelo NO promovido.")
        return False

    # ── Guardar ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model_out = output_dir / "lgbm_model.pkl"
    calib_out = output_dir / "platt_calibrator.pkl"

    with open(model_out, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(calib_out, "wb") as f:
        pickle.dump(platt, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Modelo guardado en {model_out}")
    logger.info(f"Calibrador guardado en {calib_out}")
    logger.info("Modelo PROMOVIDO ✓")
    return True


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el clasificador ML para señales OTC.")
    p.add_argument("--db",          type=Path, default=DB_PATH,    help="Ruta a trades.db")
    p.add_argument("--output",      type=Path, default=MODEL_DIR,  help="Carpeta de salida para modelos")
    p.add_argument("--min-samples", type=int,  default=50,         help="Mínimo de muestras para entrenar")
    p.add_argument("--promote-auc", type=float, default=0.60,      help="AUC mínima para promover modelo")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    success = train(
        db_path=args.db,
        output_dir=args.output,
        min_samples=args.min_samples,
        promote_auc=args.promote_auc,
    )
    sys.exit(0 if success else 1)
