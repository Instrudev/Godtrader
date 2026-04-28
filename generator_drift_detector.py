"""
generator_drift_detector.py – Detector de drift estadístico del feed OTC

El generador de precios de Exnova es interno (PRNG/simulador). Cuando Exnova
ajusta sus parámetros, las propiedades estadísticas del feed cambian. Este módulo
detecta esos cambios comparando una ventana reciente contra una ventana de referencia.

Tests estadísticos usados (sin scipy — solo numpy):
  - KS estático: max diferencia entre CDFs empíricas → sensible a cambios en distribución
  - Z-test de medias: detecta cambios en la media de retornos
  - F-ratio de varianzas: detecta cambios en volatilidad estructural
  - Autocorrelación lag-1: detecta cambios en la dependencia serial

Flujo:
  1. DriftDetector.analyze(candles, asset) → compara ventana actual vs referencia
  2. Si drift detectado → persiste alerta en drift_state.json
  3. drift_filter() en regime_filter.py consulta has_drift(asset, days=7)

Uso:
    detector = DriftDetector.load()
    result = detector.analyze(candles, "EURUSD-OTC")
    print(result)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_PATH = Path("drift_state.json")

# ─── Umbrales de significancia ────────────────────────────────────────────────

KS_THRESHOLD          = 0.10    # KS-stat máximo sin alarma (análogo a p≈0.05 con n=200)
Z_THRESHOLD           = 2.58    # z-score de media (α=0.01)
F_RATIO_THRESHOLD     = 2.0     # ratio de varianzas máximo aceptable
AUTOCORR_DELTA        = 0.25    # diferencia máxima de autocorrelación lag-1
MIN_WINDOW_SIZE       = 100     # mínimo de retornos para hacer análisis válido
REFERENCE_WINDOW_SIZE = 200     # retornos en la ventana de referencia
CURRENT_WINDOW_SIZE   = 100     # retornos en la ventana actual (comparar vs referencia)


# ─── Resultado de análisis ────────────────────────────────────────────────────

@dataclass
class DriftResult:
    asset: str
    timestamp: str
    drift_detected: bool
    ks_stat: float       = 0.0
    z_score: float       = 0.0
    f_ratio: float       = 0.0
    autocorr_delta: float = 0.0
    triggers: List[str]  = field(default_factory=list)
    details: str         = ""

    def __str__(self) -> str:
        status = "⚠ DRIFT DETECTADO" if self.drift_detected else "✓ Sin drift"
        return (
            f"[DRIFT] {self.asset} @ {self.timestamp[:16]} UTC | {status}\n"
            f"  KS={self.ks_stat:.3f} | Z={self.z_score:.2f} | "
            f"F-ratio={self.f_ratio:.2f} | ΔAutoCorr={self.autocorr_delta:.3f}\n"
            f"  Triggers: {self.triggers or 'ninguno'}"
        )


# ─── Estado persistido ────────────────────────────────────────────────────────

@dataclass
class DriftState:
    """Estado persistido entre sesiones por activo."""
    # Referencia estadística del feed (de la última ventana de referencia)
    ref_returns:    Dict[str, List[float]] = field(default_factory=dict)  # asset → list
    ref_timestamp:  Dict[str, str]         = field(default_factory=dict)  # asset → ISO ts
    # Historial de alertas de drift
    drift_alerts:   List[Dict]             = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "DriftState":
        return cls(
            ref_returns   = d.get("ref_returns",   {}),
            ref_timestamp = d.get("ref_timestamp", {}),
            drift_alerts  = d.get("drift_alerts",  []),
        )


# ─── Clase principal ──────────────────────────────────────────────────────────

class DriftDetector:
    """
    Detecta cambios estadísticos en el feed OTC comparando ventanas de retornos.

    Instancia única cargada desde disco. Uso:
        detector = DriftDetector.load()
        result = detector.analyze(candles, "EURUSD-OTC")
    """

    def __init__(self, state: Optional[DriftState] = None, path: Path = STATE_PATH) -> None:
        self.state = state or DriftState()
        self.path  = path

    # ─── Persistencia ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "DriftDetector":
        """Carga el estado desde disco, o crea uno nuevo si no existe."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(state=DriftState.from_dict(data), path=path)
            except Exception as e:
                logger.warning(f"Error cargando drift_state.json: {e}. Empezando desde cero.")
        return cls(state=DriftState(), path=path)

    def save(self) -> None:
        """Persiste el estado a disco."""
        self.path.write_text(json.dumps(self.state.to_dict(), indent=2))

    # ─── Análisis de drift ────────────────────────────────────────────────────

    def analyze(
        self,
        candles: List[Dict],
        asset: str,
        force: bool = False,
    ) -> Optional[DriftResult]:
        """
        Compara la ventana actual de retornos contra la referencia almacenada.

        Args:
            candles: Lista de dicts OHLCV recientes.
            asset:   Nombre del activo.
            force:   Si True, actualiza la referencia al terminar (rotación de ventana).

        Returns:
            DriftResult con los tests estadísticos, o None si hay datos insuficientes.
        """
        returns = _candles_to_returns(candles)
        if len(returns) < MIN_WINDOW_SIZE:
            logger.debug(f"[DRIFT] {asset}: datos insuficientes ({len(returns)} retornos < {MIN_WINDOW_SIZE})")
            return None

        now_ts = datetime.now(timezone.utc).isoformat()

        # Sin referencia previa → establecer y salir (primera vez)
        if asset not in self.state.ref_returns or len(self.state.ref_returns[asset]) < MIN_WINDOW_SIZE:
            ref_sample = returns[-REFERENCE_WINDOW_SIZE:].tolist()
            self.state.ref_returns[asset]   = ref_sample
            self.state.ref_timestamp[asset] = now_ts
            self.save()
            logger.info(f"[DRIFT] {asset}: referencia inicial establecida ({len(ref_sample)} retornos)")
            return DriftResult(asset=asset, timestamp=now_ts, drift_detected=False,
                               details="Referencia inicial establecida")

        ref = np.array(self.state.ref_returns[asset])
        cur = returns[-CURRENT_WINDOW_SIZE:]

        # ── Tests estadísticos ────────────────────────────────────────────────
        ks_stat        = _ks_statistic(ref, cur)
        z_score        = _z_test_means(ref, cur)
        f_ratio        = _f_ratio_variances(ref, cur)
        autocorr_delta = _autocorr_delta(ref, cur)

        triggers: List[str] = []
        if ks_stat > KS_THRESHOLD:
            triggers.append(f"KS={ks_stat:.3f}>{KS_THRESHOLD}")
        if abs(z_score) > Z_THRESHOLD:
            triggers.append(f"Z={z_score:.2f}")
        if f_ratio > F_RATIO_THRESHOLD or f_ratio < 1 / F_RATIO_THRESHOLD:
            triggers.append(f"F={f_ratio:.2f}")
        if abs(autocorr_delta) > AUTOCORR_DELTA:
            triggers.append(f"ΔAutoCorr={autocorr_delta:.3f}")

        drift = len(triggers) >= 2   # al menos 2 tests fallan → drift confirmado

        result = DriftResult(
            asset          = asset,
            timestamp      = now_ts,
            drift_detected = drift,
            ks_stat        = float(ks_stat),
            z_score        = float(z_score),
            f_ratio        = float(f_ratio),
            autocorr_delta = float(autocorr_delta),
            triggers       = triggers,
            details        = f"Triggers: {triggers}" if triggers else "Todos los tests OK",
        )

        logger.info(str(result))

        if drift:
            alert = {
                "asset":     asset,
                "timestamp": now_ts,
                "triggers":  triggers,
                "ks_stat":   float(ks_stat),
            }
            self.state.drift_alerts.append(alert)
            self.state.drift_alerts = self.state.drift_alerts[-50:]   # max 50 alertas
            self.save()
            logger.warning(
                f"[DRIFT] ⚠ Drift detectado en {asset}. "
                "Exnova probablemente cambió su algoritmo. Bot auto-detenido."
            )
        elif force:
            # Rotar referencia con los datos actuales si no hay drift
            self.state.ref_returns[asset]   = returns[-REFERENCE_WINDOW_SIZE:].tolist()
            self.state.ref_timestamp[asset] = now_ts
            self.save()
            logger.info(f"[DRIFT] {asset}: referencia rotada (force=True)")

        return result

    # ─── Consulta de estado ───────────────────────────────────────────────────

    def has_drift(self, asset: str, days: int = 7) -> bool:
        """
        Devuelve True si hay una alerta de drift para el activo
        en los últimos `days` días.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return any(
            a["asset"] == asset and a["timestamp"] >= cutoff
            for a in self.state.drift_alerts
        )

    def clear_drift(self, asset: str) -> None:
        """Limpia alertas de drift para un activo (tras revisión manual)."""
        self.state.drift_alerts = [
            a for a in self.state.drift_alerts if a["asset"] != asset
        ]
        self.save()
        logger.info(f"[DRIFT] Alertas de drift eliminadas para {asset}")

    def get_summary(self) -> Dict:
        """Resumen del estado actual del detector."""
        return {
            "assets_with_reference": list(self.state.ref_returns.keys()),
            "recent_alerts": [
                a for a in self.state.drift_alerts
                if a["timestamp"] >= (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            ],
            "total_alerts": len(self.state.drift_alerts),
        }


# ─── Tests estadísticos (numpy puro, sin scipy) ───────────────────────────────

def _candles_to_returns(candles: List[Dict]) -> np.ndarray:
    """
    Calcula retornos logarítmicos de cierre a cierre.
    Retornos log son estacionarios y más apropiados para tests estadísticos.
    """
    closes = np.array([float(c["close"]) for c in candles if c.get("close", 0) > 0])
    if len(closes) < 2:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(closes))
    return returns[np.isfinite(returns)]


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov estático: máxima diferencia absoluta entre CDFs empíricas.
    No requiere scipy. Aproximación: si D > 1.36/sqrt(n_eff) → p < 0.05.

    Retorna el estadístico D (0 = distribuciones idénticas, 1 = completamente distintas).
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    # Unificar todos los valores para evaluar las CDFs en los mismos puntos
    all_values = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, all_values, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_values, side="right") / len(b_sorted)

    return float(np.max(np.abs(cdf_a - cdf_b)))


def _z_test_means(ref: np.ndarray, cur: np.ndarray) -> float:
    """
    Z-test de diferencia de medias asumiendo varianzas conocidas ≈ varianzas muestrales.
    z = (mean_cur - mean_ref) / sqrt(var_ref/n_ref + var_cur/n_cur)
    """
    n_r, n_c = len(ref), len(cur)
    if n_r < 2 or n_c < 2:
        return 0.0

    var_r = float(np.var(ref, ddof=1))
    var_c = float(np.var(cur, ddof=1))
    se = np.sqrt(var_r / n_r + var_c / n_c)

    if se < 1e-12:
        return 0.0

    return float((np.mean(cur) - np.mean(ref)) / se)


def _f_ratio_variances(ref: np.ndarray, cur: np.ndarray) -> float:
    """
    F-ratio de varianzas: var_cur / var_ref.
    > F_RATIO_THRESHOLD o < 1/F_RATIO_THRESHOLD → cambio de volatilidad.
    """
    if len(ref) < 2 or len(cur) < 2:
        return 1.0

    var_r = float(np.var(ref, ddof=1))
    var_c = float(np.var(cur, ddof=1))

    if var_r < 1e-12:
        return 1.0

    return float(var_c / var_r)


def _autocorr_delta(ref: np.ndarray, cur: np.ndarray, lag: int = 1) -> float:
    """
    Diferencia de autocorrelación lag-1 entre ventana actual y referencia.
    Un cambio grande indica que la dependencia serial del generador cambió.
    """
    def autocorr(x: np.ndarray, lag: int) -> float:
        if len(x) <= lag:
            return 0.0
        x_centered = x - x.mean()
        var = np.var(x_centered)
        if var < 1e-12:
            return 0.0
        return float(np.mean(x_centered[:-lag] * x_centered[lag:]) / var)

    return autocorr(cur, lag) - autocorr(ref, lag)
