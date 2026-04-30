"""
ml_classifier.py – Clasificador ML para señales OTC.

Reemplaza al LLM (ai_brain.py) como motor de decisión.

Interfaz pública:
    ml_classifier.predict_proba_from_df(df, payout, winrate_hour)
        -> {"call_proba": float, "put_proba": float}

Modelo: LightGBM binario (WIN=1 / LOSS=0) con Platt scaling (LogisticRegression
calibrada en el set de validación).

Cuando no hay modelo entrenado → devuelve {"call_proba": 0.5, "put_proba": 0.5}
(fail-open: el pipeline upstream decide si operar o no).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ml_drift_detector import drift_detector

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

MODEL_DIR        = Path("models")
MODEL_PATH       = MODEL_DIR / "lgbm_model.pkl"
CALIBRATOR_PATH  = MODEL_DIR / "platt_calibrator.pkl"

# ─── Feature set ─────────────────────────────────────────────────────────────
# Orden fijo — NUNCA cambiar sin reentrenar. El orden debe coincidir con
# el orden de columnas usado en train_model.py.

FEATURE_COLS: List[str] = [
    # ── Indicadores técnicos clásicos ────────────────────────────────────────
    "rsi",            # RSI(14)
    "bb_pct_b",       # posición en banda Bollinger [0–1]
    "bb_width_pct",   # ancho relativo de BB
    "rel_atr",        # ATR relativo a su media reciente
    "vol_rel",        # volumen relativo (proxy OTC)
    "hour_utc",       # hora UTC [0-23]
    "weekday",        # día de semana [0-6]
    "direction",      # 1=CALL, 0=PUT
    "streak_length",  # longitud de racha actual
    "streak_pct",     # percentil histórico de la racha
    "ret_3",          # retorno acumulado 3 velas
    "ret_5",          # retorno acumulado 5 velas
    "ret_10",         # retorno acumulado 10 velas
    "autocorr_10",    # autocorrelación lag-1 retornos (ventana 10)
    "half_life",      # half-life OU (velas; cap 100 para normalizar)
    "payout",         # payout ofrecido por el broker
    "winrate_hour",   # winrate histórico del par en esa hora (0-1)
    "expiry_min",     # tiempo de expiración de la orden en minutos
    # ── Features PRNG (detección de sesgos del generador OTC) ────────────────
    "prng_last_digit_entropy",   # entropía último dígito [0-1]; 1=uniforme
    "prng_last_digit_mode_freq", # frecuencia dígito más común; esperado ~0.10
    "prng_permutation_entropy",  # complejidad ordinal [0-1]; <0.95=estructura
    "prng_runs_test_z",          # z-score test de rachas; !=0 = dependencia
    "prng_transition_entropy",   # entropía transiciones Markov [0-1]
    "prng_hurst_exponent",       # exponente de Hurst; 0.5=random, <0.5=MR
    "prng_turning_point_ratio",  # ratio puntos de giro; esperado ~0.667
    "prng_autocorr_lag2",        # autocorrelación lag-2 retornos
    "prng_autocorr_lag5",        # autocorrelación lag-5 retornos
]

_NEUTRAL = {"call_proba": 0.5, "put_proba": 0.5}


# ─── Extracción de features desde DataFrame ───────────────────────────────────

def extract_features(
    df: pd.DataFrame,
    payout: float = 0.80,
    winrate_hour: float = 0.5,
    direction: Optional[str] = None,
    expiry_min: float = 2.0,
) -> Optional[Dict[str, float]]:
    """
    Extrae el vector de features a partir del DataFrame enriquecido.

    Args:
        df:           DataFrame producido por build_dataframe() (Phase 3).
        payout:       Payout actual del par.
        winrate_hour: Winrate histórico del par en la hora actual [0-1].
        direction:    "CALL" o "PUT" (None si aún no se ha decidido).
        expiry_min:   Tiempo de expiración en minutos.

    Returns:
        Dict de features o None si los datos son insuficientes.
    """
    if df is None or df.empty or len(df) < 20:
        return None

    try:
        from indicators import (
            estimate_half_life,
            get_streak_info,
            calculate_relative_atr,
            compute_prng_features,
        )

        last = df.iloc[-1]
        closes = df["close"].astype(float).values

        # ── Features básicas del DataFrame ───────────────────────────────────
        rsi        = float(last.get("rsi",        50.0) or 50.0)
        bb_pct_b   = float(last.get("bb_pct_b",   0.5)  or 0.5)
        bb_width   = float(last.get("bb_width_pct", 1.0) or 1.0)
        vol_rel    = float(last.get("vol_rel",     1.0)  or 1.0)
        hour_utc   = int(last.get("hour_utc",   0) or 0)
        weekday    = int(last.get("weekday",    0) or 0)

        rel_atr = float(last.get("rel_atr", 1.0) or 1.0)
        if rel_atr == 1.0 and "rel_atr" not in df.columns:
            rel_atr = calculate_relative_atr(df)

        # ── Retornos acumulados ───────────────────────────────────────────────
        def cum_ret(n: int) -> float:
            if len(closes) < n + 1:
                return 0.0
            base = closes[-(n + 1)]
            if base == 0:
                return 0.0
            return float((closes[-1] - base) / base)

        ret_3  = cum_ret(3)
        ret_5  = cum_ret(5)
        ret_10 = cum_ret(10)

        # ── Autocorrelación lag-1 (ventana 10) ───────────────────────────────
        autocorr_10 = 0.0
        if len(closes) >= 12:
            window  = closes[-11:]
            returns = np.diff(window)
            if len(returns) >= 2:
                mean_r = returns.mean()
                demeaned = returns - mean_r
                denom = float(np.dot(demeaned, demeaned))
                if denom > 1e-12:
                    autocorr_10 = float(np.dot(demeaned[:-1], demeaned[1:]) / denom)

        # ── Half-life OU (cap en 100 para normalizar el rango) ────────────────
        hl = estimate_half_life(df)
        half_life = min(float(hl) if np.isfinite(hl) else 100.0, 100.0)

        # ── Racha ─────────────────────────────────────────────────────────────
        streak = get_streak_info(df)
        streak_length = float(streak.get("current_length", 0) or 0)
        streak_pct    = float(streak.get("percentile",     50.0) or 50.0)

        # ── Dirección ─────────────────────────────────────────────────────────
        dir_val = 1.0 if direction == "CALL" else (0.0 if direction == "PUT" else 0.5)

        # ── Features PRNG ─────────────────────────────────────────────────────
        prng = compute_prng_features(df)

        return {
            "rsi":          rsi,
            "bb_pct_b":     bb_pct_b,
            "bb_width_pct": bb_width,
            "rel_atr":      rel_atr,
            "vol_rel":      vol_rel,
            "hour_utc":     float(hour_utc),
            "weekday":      float(weekday),
            "direction":    dir_val,
            "streak_length": streak_length,
            "streak_pct":   streak_pct,
            "ret_3":        ret_3,
            "ret_5":        ret_5,
            "ret_10":       ret_10,
            "autocorr_10":  autocorr_10,
            "half_life":    half_life,
            "payout":       float(payout),
            "winrate_hour": float(winrate_hour),
            "expiry_min":   float(expiry_min),
            **prng,
        }

    except Exception as exc:
        logger.warning(f"extract_features error: {exc}")
        return None


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convierte el dict de features a un array 1D en el orden de FEATURE_COLS."""
    return np.array([features.get(col, 0.0) for col in FEATURE_COLS], dtype=np.float32)


# ─── Clasificador ─────────────────────────────────────────────────────────────

class MLClassifier:
    """
    Wrapper del modelo LightGBM + calibrador Platt.

    Carga explícita via load() o load_with_verification().
    Auto-carga implícita bloqueada en REMEDIATION_MODE.
    Si no existe modelo guardado, devuelve probabilidades neutras (0.5).
    """

    def __init__(self) -> None:
        self._model       = None
        self._calibrator  = None
        self._loaded      = False
        self._load_attempted = False
        self._model_hash: Optional[str]  = None
        self._model_path: Optional[Path] = None
        self._predict_count: int = 0

    # ── Carga ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path) -> str:
        """Calcula SHA256 de un archivo."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def load(self, model_path: Path = MODEL_PATH, calib_path: Path = CALIBRATOR_PATH) -> bool:
        """
        Carga el modelo y calibrador desde disco con logging completo.

        Returns:
            True si la carga fue exitosa, False si falta algún archivo.
        """
        self._load_attempted = True
        if not model_path.exists():
            logger.info(f"Modelo ML no encontrado en {model_path}. Usando fallback.")
            return False
        if not calib_path.exists():
            logger.info(f"Calibrador no encontrado en {calib_path}. Usando fallback.")
            return False

        try:
            model_hash = self._sha256(model_path)
            mod_time = datetime.fromtimestamp(
                os.path.getmtime(model_path), tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")

            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            with open(calib_path, "rb") as f:
                self._calibrator = pickle.load(f)

            self._loaded = True
            self._model_hash = model_hash
            self._model_path = model_path
            self._predict_count = 0

            # Feature importances summary
            n_features = getattr(self._model, "n_features_", "?")
            importances = getattr(self._model, "feature_importances_", [])
            nonzero = sum(1 for v in importances if v > 0)

            logger.info(
                f"[ML] Modelo cargado | path={model_path} | "
                f"hash={model_hash[:12]}... | modified={mod_time} | "
                f"features={n_features} (active={nonzero})"
            )
            return True
        except Exception as exc:
            logger.error(f"Error cargando modelo ML: {exc}")
            self._loaded = False
            return False

    def load_with_verification(
        self,
        model_path: Path = MODEL_PATH,
        calib_path: Path = CALIBRATOR_PATH,
        expected_hash: Optional[str] = None,
    ) -> bool:
        """
        Carga el modelo con verificación SHA256 opcional.

        Si expected_hash se proporciona y no coincide, la carga falla
        y se registra en logs/security_halts.log.

        Returns:
            True si la carga fue exitosa y el hash coincide (si se verificó).
        """
        if not model_path.exists():
            logger.info(f"Modelo no encontrado en {model_path}.")
            return False

        actual_hash = self._sha256(model_path)

        if expected_hash is not None and actual_hash != expected_hash:
            logger.error(
                f"[ML] HASH MISMATCH | expected={expected_hash[:12]}... "
                f"actual={actual_hash[:12]}... | path={model_path}"
            )
            try:
                from iqservice import _security_logger
                ts = datetime.now(timezone.utc).isoformat()
                _security_logger.warning(
                    f"[{ts}] HALT_TYPE=ML_MODEL_HASH_MISMATCH "
                    f"expected={expected_hash} actual={actual_hash} "
                    f"path={model_path} "
                    f"action_required=Verify model integrity before loading"
                )
            except Exception:
                pass
            return False

        return self.load(model_path, calib_path)

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_info(self) -> Dict:
        """Snapshot del modelo cargado para auditoría."""
        return {
            "loaded": self._loaded,
            "path": str(self._model_path) if self._model_path else None,
            "hash": self._model_hash,
            "predict_count": self._predict_count,
        }

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Devuelve probabilidades de éxito para CALL y PUT.

        Auto-carga implícita bloqueada en REMEDIATION_MODE.
        Si el modelo no está cargado, devuelve probabilidades neutras (0.5).

        Args:
            features: Dict con las claves de FEATURE_COLS. La clave "direction"
                      se sobreescribe internamente para calcular ambas probas.

        Returns:
            {"call_proba": float, "put_proba": float} en rango [0, 1].
        """
        if not self._loaded and not self._load_attempted:
            # Bloquear auto-carga en REMEDIATION_MODE
            try:
                from iqservice import REMEDIATION_MODE
                if REMEDIATION_MODE:
                    logger.warning(
                        "[ML] Auto-carga bloqueada en REMEDIATION_MODE. "
                        "Usar load() o load_with_verification() explícitamente."
                    )
                    self._load_attempted = True
                else:
                    self.load()
            except ImportError:
                self.load()

        if not self._loaded:
            return dict(_NEUTRAL)

        self._predict_count += 1
        if self._predict_count % 100 == 0:
            logger.info(
                f"[ML] Checkpoint: {self._predict_count} predicciones | "
                f"modelo={self._model_hash[:12] if self._model_hash else '?'}..."
            )

        try:
            # call_proba: features con direction=CALL(1)
            feat_call = dict(features)
            feat_call["direction"] = 1.0
            arr_call = features_to_array(feat_call).reshape(1, -1)

            # put_proba: features con direction=PUT(0)
            feat_put = dict(features)
            feat_put["direction"] = 0.0
            arr_put = features_to_array(feat_put).reshape(1, -1)

            best_iter = getattr(self._model, "best_iteration_", None) or getattr(self._model, "best_iteration", None)
            raw_call = self._model.predict(arr_call, num_iteration=best_iter)[0]
            raw_put  = self._model.predict(arr_put,  num_iteration=best_iter)[0]

            call_proba = float(self._calibrator.predict_proba([[raw_call]])[0][1])
            put_proba  = float(self._calibrator.predict_proba([[raw_put]])[0][1])

            # Clamp al rango [0, 1] por seguridad
            call_proba = max(0.0, min(1.0, call_proba))
            put_proba  = max(0.0, min(1.0, put_proba))

            # Registrar predicción para drift detection
            try:
                drift_detector.record_prediction(call_proba, put_proba)
            except Exception as exc:
                logger.debug(f"[ML_DRIFT] record error: {exc}")

            return {"call_proba": call_proba, "put_proba": put_proba}

        except Exception as exc:
            logger.error(f"predict_proba error: {exc}")
            return dict(_NEUTRAL)

    def predict_proba_from_df(
        self,
        df: pd.DataFrame,
        payout: float = 0.80,
        winrate_hour: float = 0.5,
        expiry_min: float = 2.0,
    ) -> Dict[str, float]:
        """
        Extrae features del DataFrame y devuelve probabilidades.

        Convenience method que combina extract_features + predict_proba.
        """
        features = extract_features(df, payout=payout, winrate_hour=winrate_hour, expiry_min=expiry_min)
        if features is None:
            return dict(_NEUTRAL)
        return self.predict_proba(features)


# ─── Singleton ────────────────────────────────────────────────────────────────

ml_classifier = MLClassifier()
