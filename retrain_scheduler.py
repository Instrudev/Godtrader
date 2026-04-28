"""
retrain_scheduler.py – Loop de reentrenamiento automático del clasificador ML.

Lógica:
  Cada N horas (default 6h) comprueba si hay suficientes trades nuevos acumulados
  desde el último entrenamiento. Si se cumplen ambas condiciones (tiempo + datos),
  reentrena y compara el challenger contra el champion actual usando Profit Factor
  sobre los trades más recientes. Solo promueve si el challenger supera al champion
  (o si no existe champion previo y el challenger cumple criterios absolutos).

Versionado:
  Cada modelo entrenado se guarda en models/versions/YYYYMMDD_HHMMSS/
  con sus métricas. El champion activo siempre está en:
    models/lgbm_model.pkl
    models/platt_calibrator.pkl
  Se mantienen las últimas MAX_VERSIONS versiones (default 5).

Uso como módulo:
    from retrain_scheduler import retrain_scheduler
    await retrain_scheduler.start()   # en lifespan FastAPI

CLI (entrenamiento único inmediato):
    python retrain_scheduler.py --now
    python retrain_scheduler.py --now --min-new-trades 30
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from database import fetch_training_data

logger = logging.getLogger(__name__)

# ─── Constantes ───────────────────────────────────────────────────────────────

MODEL_DIR        = Path("models")
VERSIONS_DIR     = MODEL_DIR / "versions"
STATE_PATH       = MODEL_DIR / "retrain_state.json"
CHAMPION_MODEL   = MODEL_DIR / "lgbm_model.pkl"
CHAMPION_CALIB   = MODEL_DIR / "platt_calibrator.pkl"

CHECK_INTERVAL_H  = 6      # horas entre comprobaciones del loop
MIN_NEW_TRADES    = 50     # mínimo de trades nuevos para disparar reentrenamiento
MIN_TOTAL_TRADES  = 50     # mínimo absoluto total para entrenar (pasado a train_model)
MAX_VERSIONS      = 5      # cuántas versiones históricas conservar
EVAL_WINDOW       = 30     # trades más recientes para comparar champion vs challenger


# ─── Estado persistente ───────────────────────────────────────────────────────

class _State:
    """Estado serializado en models/retrain_state.json."""

    def __init__(self) -> None:
        self.last_trained_at:    Optional[str] = None   # ISO-8601 UTC
        self.trades_at_last_run: int           = 0
        self.current_version:    Optional[str] = None
        self.versions:           List[Dict]    = []     # lista de {version, metrics, trained_at}

    @classmethod
    def load(cls) -> "_State":
        s = cls()
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
                s.last_trained_at    = data.get("last_trained_at")
                s.trades_at_last_run = int(data.get("trades_at_last_run", 0))
                s.current_version    = data.get("current_version")
                s.versions           = data.get("versions", [])
            except Exception as exc:
                logger.warning(f"Error cargando retrain_state.json: {exc}. Usando estado vacío.")
        return s

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(
                {
                    "last_trained_at":    self.last_trained_at,
                    "trades_at_last_run": self.trades_at_last_run,
                    "current_version":    self.current_version,
                    "versions":           self.versions,
                },
                indent=2,
            )
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version_tag() -> str:
    """Tag de versión basado en timestamp UTC: YYYYMMDD_HHMMSS."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _profit_factor_on_recent(
    state: _State,
    n: int = EVAL_WINDOW,
) -> Optional[float]:
    """
    Calcula el Profit Factor del modelo champion sobre los N trades más recientes.

    Devuelve None si no hay suficientes trades cerrados o no hay champion.
    """
    if not CHAMPION_MODEL.exists():
        return None

    try:
        from database import fetch_training_data
        from ml_classifier import MLClassifier, extract_features
        from indicators import build_dataframe

        rows = fetch_training_data()[-n:]
        if len(rows) < 10:
            return None

        clf = MLClassifier()
        clf.load()
        if not clf.is_loaded():
            return None

        wins_profit  = 0.0
        losses_value = 0.0

        for row in rows:
            # Reconstruir features desde la fila de DB
            from train_model import _row_to_features
            feat_arr = _row_to_features(row)
            if feat_arr is None:
                continue
            from ml_classifier import FEATURE_COLS, features_to_array
            feat_dict = dict(zip(FEATURE_COLS, feat_arr.tolist()))
            probas = clf.predict_proba(feat_dict)

            direction = str(row.get("direction", "CALL")).upper()
            proba = probas["call_proba"] if direction == "CALL" else probas["put_proba"]
            payout = float(row.get("payout", 0.80) or 0.80)
            result = str(row.get("result", ""))

            # Solo contar si el modelo habría operado (proba >= 0.65)
            if proba >= 0.65:
                if result == "WIN":
                    wins_profit  += payout
                elif result == "LOSS":
                    losses_value += 1.0

        if losses_value == 0:
            return float("inf") if wins_profit > 0 else None
        return wins_profit / losses_value

    except Exception as exc:
        logger.warning(f"Error calculando Profit Factor del champion: {exc}")
        return None


def _profit_factor_from_metrics(metrics: Optional[Dict]) -> float:
    """Extrae PF de las métricas guardadas del challenger. Devuelve 0 si no hay."""
    if not metrics:
        return 0.0
    return float(metrics.get("profit_factor_65", 0.0))


# ─── Entrenamiento y promoción ────────────────────────────────────────────────

def _train_challenger(version: str) -> Optional[Dict]:
    """
    Entrena un modelo challenger y lo guarda en models/versions/<version>/.

    Returns:
        Dict de métricas si el challenger cumple criterios absolutos, None si no.
    """
    from train_model import train, _walk_forward_split, _profit_factor_sim, _auc_roc, _brier_score
    from train_model import _row_to_features
    from ml_classifier import FEATURE_COLS
    import numpy as np

    version_dir = VERSIONS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Entrenar usando train_model.train con output en la versión
    promoted = train(
        output_dir=version_dir,
        min_samples=MIN_TOTAL_TRADES,
    )

    if not promoted:
        # Limpiar directorio vacío
        shutil.rmtree(version_dir, ignore_errors=True)
        return None

    # Calcular métricas para comparación
    rows = fetch_training_data()
    X_list, y_list = [], []
    for row in rows:
        arr = _row_to_features(row)
        if arr is None:
            continue
        X_list.append(arr)
        y_list.append(1.0 if row.get("result") == "WIN" else 0.0)

    if len(X_list) < 30:
        return {"profit_factor_65": 0.0, "auc": 0.0, "n": len(X_list)}

    X = np.stack(X_list)
    y = np.array(y_list)
    _, _, _, _, X_test, y_test = _walk_forward_split(X, y)

    if len(y_test) < 5:
        return {"profit_factor_65": 0.0, "auc": 0.0, "n": len(y)}

    import pickle
    model_path = version_dir / "lgbm_model.pkl"
    calib_path = version_dir / "platt_calibrator.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)

    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calib.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    auc = _auc_roc(y_test, cal_test)
    pf_65 = _profit_factor_sim(y_test, cal_test, threshold=0.65)

    metrics = {
        "profit_factor_65": round(pf_65, 4),
        "auc":               round(auc, 4),
        "n":                 len(y),
        "n_test":            len(y_test),
    }

    # Guardar métricas junto al modelo
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Challenger {version}: AUC={auc:.4f} PF@65%={pf_65:.2f} (N={len(y)})")
    return metrics


def _promote_champion(version: str) -> None:
    """
    Copia el modelo challenger al slot de champion (models/lgbm_model.pkl).
    Archiva el champion anterior en su versión si la tenía.
    """
    challenger_dir = VERSIONS_DIR / version

    shutil.copy2(challenger_dir / "lgbm_model.pkl",      CHAMPION_MODEL)
    shutil.copy2(challenger_dir / "platt_calibrator.pkl", CHAMPION_CALIB)

    logger.info(f"Champion actualizado → versión {version}")

    # Recargar el singleton global de ml_classifier
    try:
        from ml_classifier import ml_classifier
        ml_classifier.load()
        logger.info("ml_classifier singleton recargado con nuevo champion.")
    except Exception as exc:
        logger.warning(f"No se pudo recargar ml_classifier: {exc}")


def _cleanup_old_versions(keep: int = MAX_VERSIONS) -> None:
    """Elimina versiones antiguas dejando solo las `keep` más recientes."""
    if not VERSIONS_DIR.exists():
        return
    dirs = sorted(VERSIONS_DIR.iterdir(), key=lambda d: d.name)
    to_delete = dirs[:-keep] if len(dirs) > keep else []
    for d in to_delete:
        shutil.rmtree(d, ignore_errors=True)
        logger.debug(f"Versión eliminada: {d.name}")


# ─── Scheduler ────────────────────────────────────────────────────────────────

class RetrainScheduler:
    """
    Scheduler de reentrenamiento automático del clasificador ML.

    Corre como tarea asyncio en background. Cada CHECK_INTERVAL_H horas evalúa
    si debe reentrenar según datos acumulados. Promueve el nuevo modelo solo si
    supera al champion actual en Profit Factor sobre los trades más recientes.
    """

    def __init__(
        self,
        check_interval_h: float = CHECK_INTERVAL_H,
        min_new_trades: int = MIN_NEW_TRADES,
    ) -> None:
        self._interval_h    = check_interval_h
        self._min_new       = min_new_trades
        self._running       = False
        self._task: Optional[asyncio.Task] = None
        self._state         = _State.load()
        self._last_check_at: Optional[str] = None
        self._last_result:   Optional[str] = None

    # ── Control ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Inicia el loop en background. Llamar desde lifespan de FastAPI."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"RetrainScheduler iniciado | intervalo={self._interval_h}h | "
            f"min_new_trades={self._min_new}"
        )

    def stop(self) -> None:
        """Detiene el loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("RetrainScheduler detenido.")

    # ── Loop ─────────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Comprueba cada CHECK_INTERVAL_H horas si debe reentrenar."""
        while self._running:
            try:
                self._last_check_at = _now_iso()
                if self._should_retrain():
                    logger.info("[RETRAIN] Condiciones cumplidas → iniciando reentrenamiento...")
                    await asyncio.to_thread(self._run_retrain)
                else:
                    logger.debug("[RETRAIN] Condiciones no cumplidas. Siguiente check en "
                                 f"{self._interval_h}h.")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[RETRAIN] Error en loop: {exc}", exc_info=True)

            # Esperar hasta el próximo check
            try:
                await asyncio.sleep(self._interval_h * 3600)
            except asyncio.CancelledError:
                break

        self._running = False

    # ── Condición de disparo ──────────────────────────────────────────────────

    def _should_retrain(self) -> bool:
        """Devuelve True si hay suficientes trades nuevos desde el último entreno."""
        try:
            total = len(fetch_training_data())
        except Exception:
            return False

        new_since_last = total - self._state.trades_at_last_run
        logger.info(
            f"[RETRAIN] Trades totales={total} | "
            f"desde último entreno={new_since_last} (mínimo {self._min_new})"
        )
        return new_since_last >= self._min_new

    # ── Ciclo de entrenamiento + promoción ────────────────────────────────────

    def _run_retrain(self) -> None:
        """Entrena challenger, compara con champion, promueve si es mejor."""
        version = _version_tag()
        logger.info(f"[RETRAIN] Entrenando challenger versión {version}...")

        challenger_metrics = _train_challenger(version)

        if challenger_metrics is None:
            logger.warning("[RETRAIN] Challenger no cumplió criterios absolutos. No se promueve.")
            self._last_result = f"challenger_{version}: rechazado (criterios absolutos)"
            return

        # Comparar contra champion
        champion_pf = _profit_factor_on_recent(self._state, n=EVAL_WINDOW)
        challenger_pf = _profit_factor_from_metrics(challenger_metrics)

        logger.info(
            f"[RETRAIN] Champion PF@65% reciente: "
            f"{champion_pf:.3f}" if champion_pf is not None else "[RETRAIN] Champion PF@65%: N/A (sin champion)"
        )
        logger.info(f"[RETRAIN] Challenger PF@65% test: {challenger_pf:.3f}")

        should_promote = (
            champion_pf is None           # no hay champion → promover
            or challenger_pf > champion_pf  # challenger es mejor
        )

        if should_promote:
            _promote_champion(version)
            self._state.current_version = version
            self._last_result = f"promoted_{version}: PF={challenger_pf:.3f}"
            logger.info(f"[RETRAIN] ✓ Challenger {version} PROMOVIDO (PF={challenger_pf:.3f})")
        else:
            logger.info(
                f"[RETRAIN] ✗ Challenger no supera champion "
                f"({challenger_pf:.3f} ≤ {champion_pf:.3f}). Champion conservado."
            )
            self._last_result = (
                f"challenger_{version}: rechazado "
                f"(PF={challenger_pf:.3f} ≤ champion={champion_pf:.3f})"
            )

        # Actualizar estado
        try:
            total = len(fetch_training_data())
        except Exception:
            total = self._state.trades_at_last_run
        self._state.last_trained_at    = _now_iso()
        self._state.trades_at_last_run = total
        self._state.versions.append({
            "version":    version,
            "trained_at": _now_iso(),
            "metrics":    challenger_metrics,
            "promoted":   should_promote,
        })
        self._state.save()
        _cleanup_old_versions(keep=MAX_VERSIONS)

    # ── API de estado ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Dict para el endpoint GET /retrain/status."""
        return {
            "running":           self._running,
            "check_interval_h":  self._interval_h,
            "min_new_trades":    self._min_new,
            "last_check_at":     self._last_check_at,
            "last_result":       self._last_result,
            "last_trained_at":   self._state.last_trained_at,
            "trades_at_last_run": self._state.trades_at_last_run,
            "current_version":   self._state.current_version,
            "versions":          self._state.versions[-5:],
        }

    def trigger_now(self) -> str:
        """Fuerza un ciclo de reentrenamiento inmediato (sin esperar intervalo)."""
        if not self._running:
            return "Scheduler no está corriendo."
        asyncio.create_task(asyncio.to_thread(self._run_retrain))
        return "Reentrenamiento iniciado en background."


# ─── Singleton ────────────────────────────────────────────────────────────────

retrain_scheduler = RetrainScheduler()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Reentrenamiento del clasificador ML OTC.")
    p.add_argument("--now",            action="store_true", help="Forzar entrenamiento inmediato")
    p.add_argument("--min-new-trades", type=int, default=MIN_NEW_TRADES,
                   help="Mínimo de trades nuevos para reentrenar")
    p.add_argument("--force",          action="store_true",
                   help="Ignorar condición de trades nuevos y reentrenar directamente")
    args = p.parse_args()

    if args.now or args.force:
        sched = RetrainScheduler(min_new_trades=args.min_new_trades)
        if args.force or sched._should_retrain():
            sched._run_retrain()
            print(f"\nResultado: {sched._last_result}")
        else:
            from database import fetch_training_data
            total = len(fetch_training_data())
            new = total - sched._state.trades_at_last_run
            print(
                f"Condición no cumplida: {new} trades nuevos < {args.min_new_trades} requeridos.\n"
                "Usa --force para ignorar la condición."
            )
        sys.exit(0)

    print("Uso: python retrain_scheduler.py --now  (para entrenamiento inmediato)")
    print("     python retrain_scheduler.py --force (forzar sin verificar condición)")
