"""
iqservice.py – Wrapper de la API de IQ Option
Gestiona conexión, listado de activos, datos históricos,
stream en tiempo real y ejecución de órdenes.
"""

import inspect
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from iqoptionapi.stable_api import IQ_Option
from iqoptionapi.api import IQOptionAPI
import iqoptionapi.global_value as global_value

logger = logging.getLogger(__name__)

# ─── Configuración de Remediación ────────────────────────────────────────────
# Estas constantes controlan el halt operativo durante la fase de remediación.
# Para salir del modo remediación, ver CHANGELOG_REMEDIATION.md.

REMEDIATION_MODE: bool = True       # True = branch de remediación activa
FORCE_DEMO_ACCOUNT: bool = True     # True = bloquear ejecución en cuenta real
ALLOWED_ACCOUNT_TYPES = {"PRACTICE"}  # Whitelist de tipos de cuenta permitidos
ALLOW_DEPRECATED_TRADERS: bool = False  # True = permitir import de trader/paper_trader/ai_brain

# ─── Modo sin ML (Tarea 1.6) ────────────────────────────────────────────────
# Activado tras hallazgos de Tarea 1.5: modelo ML no añade valor predictivo real.
# Temporal hasta reentrenamiento con AUC ≥ 0.62 (Tarea 3.1).

ML_DISABLED_MODE: bool = True              # True = operar sin ML gate
ML_DISABLED_MIN_SCORE: float = 0.75        # Score mínimo de estrategia sin ML
ML_DISABLED_MAX_ASSET_LOSSES: int = 2      # Pérdidas máx por activo (vs 3 normal)
ML_DISABLED_MAX_DAILY_LOSSES: int = 4      # Pérdidas máx globales (vs 6 normal)
ML_DISABLED_MAX_DAILY_TRADES: int = 5      # Trades máx por día (vs 15 normal)

# ─── Logger de seguridad dedicado ────────────────────────────────────────────

_security_logger = logging.getLogger("security_halts")
_security_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_security_log_dir, exist_ok=True)
_security_handler = logging.FileHandler(
    os.path.join(_security_log_dir, "security_halts.log"), encoding="utf-8"
)
_security_handler.setFormatter(
    logging.Formatter("%(message)s")
)
_security_logger.addHandler(_security_handler)
_security_logger.setLevel(logging.WARNING)

class Exnova_Option(IQ_Option):
    """
    Subclase que intercepta la conexión nativa de IQ_Option
    para forzar el endpoint hacia exnova.com dinámicamente.
    """
    def __init__(self, email, password):
        super().__init__(email, password)

    def connect(self):
        try:
            self.api.close()
        except Exception:
            pass

        # Forzar host principal a trade.exnova.com
        self.api = IQOptionAPI("trade.exnova.com", self.email, self.password)
        # Forzar socket a ws.trade.exnova.com
        self.api.wss_url = "wss://ws.trade.exnova.com/echo/websocket"
        
        # Monkey-patch dinámico para suplantar URLs hardcodeadas
        original_send_v2 = self.api.send_http_request_v2
        def patched_send_v2(*args, **kwargs):
            url = kwargs.get('url', args[0] if args else "")
            
            # Reemplazos específicos de dominio Exnova
            if "auth.iqoption.com/api" in url:
                url = url.replace("auth.iqoption.com/api", "api.trade.exnova.com")
            elif "iqoption.com" in url:
                url = url.replace("iqoption.com", "trade.exnova.com")
                
            if 'url' in kwargs:
                kwargs['url'] = url
            elif args:
                args = (url,) + args[1:]
                
            # Inyectar validación de origen en headers si es login
            if "api.trade.exnova.com/v2/login" in url:
                headers = kwargs.get('headers') or {}
                headers["Origin"] = "https://trade.exnova.com"
                headers["Referer"] = "https://trade.exnova.com/"
                kwargs['headers'] = headers

            return original_send_v2(*args, **kwargs)
        self.api.send_http_request_v2 = patched_send_v2

        self.api.set_session(headers=self.SESSION_HEADER, cookies=self.SESSION_COOKIE)
        check, reason = self.api.connect()

        if check:
            self.re_subscribe_stream()
            _timeout = time.time() + 30
            while global_value.balance_id is None:
                if time.time() > _timeout:
                    logger.error("Timeout (30s) esperando balance_id de Exnova.")
                    return False, "balance_id timeout"
                time.sleep(0.1)
            self.position_change_all("subscribeMessage", global_value.balance_id)
            self.order_changed_all("subscribeMessage")
            self.api.setOptions(1, True)
            return True, None
        else:
            return False, reason

class IQService:
    """
    Capa de servicio sobre iqoptionapi.
    Expone métodos de alto nivel para el bot de trading.
    """

    def __init__(self):
        self.api: Optional[IQ_Option] = None
        self.connected: bool          = False

    # ─── Conexión ─────────────────────────────────────────────────────────────

    def connect(self, email: str, password: str) -> Tuple[bool, str]:
        """
        Establece conexión con Exnova dinámicamente.

        El bot operará en modo PRÁCTICA por defecto.
        Tras conectar, sincroniza el diccionario de activos de la librería
        con todos los activos disponibles en el broker.
        """
        self.api = Exnova_Option(email, password)
        check, reason = self.api.connect()
        if check:
            self.connected = True
            self.api.change_balance('PRACTICE')  # Asegurado: siempre operará en práctica
            self._sync_actives()
            self._log_startup_banner(email)
        else:
            logger.error(f"Fallo de conexión a Exnova: {reason}")
        return check, reason

    def _log_startup_banner(self, email: str) -> None:
        """Loguea banner de seguridad con estado de remediación y tipo de cuenta."""
        try:
            account_type = self.get_account_type()
        except Exception:
            account_type = "UNKNOWN"

        mode_label = "REMEDIATION" if REMEDIATION_MODE else "PRODUCTION"
        guard_label = "ACTIVE" if FORCE_DEMO_ACCOUNT else "DISABLED"
        match = account_type in ALLOWED_ACCOUNT_TYPES

        banner = (
            f"\n{'=' * 60}\n"
            f"  STARTUP SECURITY CHECK\n"
            f"  Mode:           {mode_label}\n"
            f"  Demo Guard:     {guard_label}\n"
            f"  Account Type:   {account_type}\n"
            f"  Allowed Types:  {ALLOWED_ACCOUNT_TYPES}\n"
            f"  Config Match:   {'OK' if match else 'MISMATCH — ORDERS WILL BE BLOCKED'}\n"
            f"  User:           {email}\n"
            f"{'=' * 60}"
        )

        if REMEDIATION_MODE:
            logger.warning(f"REMEDIATION MODE — DEMO ONLY{banner}")
        else:
            logger.info(f"Production mode{banner}")

    def _sync_actives(self) -> None:
        """
        Inyecta activos del broker en el diccionario de la librería iqoptionapi.
        Sin esto, get_candles() falla para activos nuevos (crypto OTC, commodities, etc.)
        que no están en el constants.ACTIVES hardcodeado de la librería.
        """
        try:
            import iqoptionapi.constants as OP_code
            binary_data = self.api.get_all_init_v2()
            injected = 0
            for opt_type in ("binary", "turbo"):
                actives = binary_data.get(opt_type, {}).get("actives", {})
                for active_id, active in actives.items():
                    raw_name = str(active.get("name", ""))
                    name = raw_name.split(".")[-1] if "." in raw_name else raw_name
                    if name and name not in OP_code.ACTIVES:
                        try:
                            OP_code.ACTIVES[name] = int(active_id)
                            injected += 1
                        except (ValueError, TypeError):
                            pass
            if injected:
                logger.info(f"Sincronizados {injected} activos nuevos del broker (total: {len(OP_code.ACTIVES)})")
        except Exception as e:
            logger.warning(f"Error sincronizando activos: {e}")

    def get_account_type(self) -> str:
        """
        Consulta al broker el tipo de cuenta activo (PRACTICE o REAL).
        No depende del monkey-patch de change_balance — lee el estado real.

        Returns:
            "PRACTICE" o "REAL" según la respuesta del broker.

        Raises:
            RuntimeError: Si no hay conexión o la respuesta es inesperada.
        """
        if self.api is None:
            raise RuntimeError("API not initialized — cannot determine account type")
        try:
            balance_mode = self.api.get_balance_mode()
            if balance_mode is None:
                raise RuntimeError("Broker returned None for balance mode")
            return str(balance_mode).upper()
        except AttributeError:
            raise RuntimeError("API does not support get_balance_mode()")

    def is_connected(self) -> bool:
        """Verifica de forma segura si la conexión sigue viva. Si no, intenta reconectar."""
        if not self.connected or self.api is None:
            return False
        try:
            # check_connect revisa el estado del websocket en la librería
            if hasattr(self.api, "check_connect"):
                is_conn = bool(self.api.check_connect())
                if not is_conn:
                    logger.warning("Conexión websocket caída. Intentando reconectar automáticamente...")
                    check, _ = self.api.connect()
                    if check:
                        logger.info("Reconexión automática exitosa.")
                    else:
                        logger.error("Falló la reconexión automática.")
                    return bool(check)
                return True
            return True
        except Exception:
            return False

    # ─── Listado de Activos ───────────────────────────────────────────────────

    def get_assets(self) -> List[Dict]:
        """
        Retorna todos los activos disponibles con su estado (abierto/cerrado).
        Filtra y oculta aquellos no soportados por el diccionario local de la librería.
        """
        if not self.is_connected():
            return []

        # Extraer nombres soportados
        try:
            from iqoptionapi.constants import ACTIVES as SUPPORTED_ACTIVES
            supported_names = set(SUPPORTED_ACTIVES.keys())
        except ImportError:
            supported_names = set()

        assets = []
        now    = time.time()

        # ── Binary y Turbo ────────────────────────────────────────────────────
        try:
            binary_data = self.api.get_all_init_v2()
            for opt_type in ("binary", "turbo"):
                actives = binary_data.get(opt_type, {}).get("actives", {})
                for _, active in actives.items():
                    raw_name = str(active.get("name", ""))
                    name = raw_name.split(".")[-1] if "." in raw_name else raw_name
                    if not name:
                        continue
                    is_open = (
                        active.get("enabled", False)
                        and not active.get("is_suspended", False)
                    )
                    assets.append({"name": name, "type": opt_type, "id": name, "open": is_open})
        except Exception as e:
            logger.error(f"Error binary/turbo: {e}")

        # ── Digital ───────────────────────────────────────────────────────────
        try:
            digital_raw = self.api.get_digital_underlying_list_data()
            if digital_raw and "underlying" in digital_raw:
                for item in digital_raw["underlying"]:
                    name = item.get("underlying", "")
                    if not name or (supported_names and name not in supported_names):
                        continue
                    is_open = any(
                        s["open"] < now < s["close"] for s in item.get("schedule", [])
                    )
                    assets.append({"name": name, "type": "digital", "id": name, "open": is_open})
        except Exception as e:
            logger.error(f"Error digital: {e}")

        # ── Forex / Crypto / CFD ──────────────────────────────────────────────
        for instr_type in ("forex", "crypto", "cfd"):
            try:
                ins_data    = self.api.get_instruments(instr_type)
                instruments = ins_data.get("instruments", [])
                for detail in instruments:
                    name = detail.get("name", "")
                    if not name or (supported_names and name not in supported_names):
                        continue
                    is_open = any(
                        s["open"] < now < s["close"] for s in detail.get("schedule", [])
                    )
                    assets.append({"name": name, "type": instr_type, "id": name, "open": is_open})
            except Exception as e:
                logger.error(f"Error {instr_type}: {e}")

        assets.sort(key=lambda x: (not x["open"], x["name"]))
        return assets

    # ─── Datos Históricos ─────────────────────────────────────────────────────

    def get_candles(
        self, asset_name: str, interval: int = 60, count: int = 300
    ) -> List[Dict]:
        """
        Obtiene velas históricas cerradas.

        Args:
            asset_name: Nombre del activo (e.g., "EURUSD")
            interval:   Segundos por vela
            count:      Cantidad de velas (recomendado 300 para EMA200)

        Returns:
            Lista de dicts: {time, open, high, low, close, volume}
        """
        if not self.is_connected():
            return []

        try:
            # Pedir count + 1 para asegurar que al descartar la vela en formación nos queden las requeridas
            raw = self.api.get_candles(asset_name, interval, count + 1, time.time())
            if not raw:
                # Stealth disconnect: A veces check_connect es True pero get_candles falla
                logger.warning("get_candles devolvió vacío. Forzando reconexión...")
                self.api.connect()
                time.sleep(1.0)
                raw = self.api.get_candles(asset_name, interval, count + 1, time.time())
                if not raw:
                    return []
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

        candles = []
        now = time.time()
        for c in raw:
            from_ts = int(c.get("from", 0))
            
            # Ignorar la vela si todavía se está formando
            if from_ts + interval > now:
                continue

            candle: Dict = {
                "time":  from_ts,
                "open":  float(c.get("open", 0)),
                "high":  float(c.get("max", 0)),
                "low":   float(c.get("min", 0)),
                "close": float(c.get("close", 0)),
            }
            # Incluir volumen si está disponible en la respuesta
            if "volume" in c:
                candle["volume"] = float(c["volume"])
            candles.append(candle)

        return candles[-count:]  # Retornar exactamente el count solicitado

    # ─── Stream en Tiempo Real ────────────────────────────────────────────────

    def start_stream(self, asset_name: str, interval: int = 60) -> bool:
        """Inicia el stream de velas en tiempo real."""
        if not self.is_connected():
            return False
        try:
            self.api.start_candles_stream(asset_name, interval, 100)
            logger.debug(f"Stream iniciado: {asset_name} ({interval}s)")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar stream {asset_name}: {e}")
            return False

    def get_realtime_candle(
        self, asset_name: str, interval: int = 60
    ) -> Optional[Dict]:
        """
        Retorna la vela que se está formando actualmente.
        Prioriza el timestamp de la vela actual; fallback a la más reciente.
        """
        if not self.is_connected():
            return None
        try:
            candles = self.api.get_realtime_candles(asset_name, interval)
            if not candles:
                return None
            current_ts = int(time.time() // interval * interval)
            c = candles.get(current_ts) or candles.get(max(candles.keys()))
            if not c:
                return None
            return {
                "time":  int(c["from"]),
                "open":  float(c["open"]),
                "high":  float(c["max"]),
                "low":   float(c["min"]),
                "close": float(c["close"]),
            }
        except Exception as e:
            logger.error(f"get_realtime_candle: {e}")
            return None

    def stop_stream(self, asset_name: str, interval: int = 60) -> None:
        """Detiene el stream de velas en tiempo real."""
        if not self.is_connected() or not asset_name:
            return
        try:
            self.api.stop_candles_stream(asset_name, interval)
        except Exception as e:
            logger.error(f"Error al detener stream: {e}")

    # ─── Payout actual del par ────────────────────────────────────────────────

    def get_payout(self, asset: str, asset_type: str = "binary") -> Optional[float]:
        """
        Intenta obtener el payout actual del par en Exnova.

        Devuelve un valor entre 0 y 1 (e.g. 0.80 = 80%), o None si no está disponible.
        La ausencia de payout no debe bloquear el bot (fail-open en regime_filter).
        """
        if not self.is_connected():
            return None
        try:
            data = self.api.get_all_init_v2()
            opt_type = "turbo" if asset_type == "turbo" else "binary"
            actives = data.get(opt_type, {}).get("actives", {})
            for _, active in actives.items():
                raw_name = str(active.get("name", ""))
                name = raw_name.split(".")[-1] if "." in raw_name else raw_name
                if name == asset:
                    # La API devuelve commission = % que retiene el broker.
                    # Payout real del trader = 1 - commission/100
                    profit = active.get("option", {}).get("profit", {})
                    commission = profit.get("commission")
                    if commission is not None:
                        return 1.0 - float(commission) / 100.0
        except Exception as e:
            logger.debug(f"get_payout: no se pudo obtener payout para {asset}: {e}")
        return None

    # ─── Ejecución de Órdenes ─────────────────────────────────────────────────

    def _enforce_demo_guard(self, asset: str, direction: str, caller: str) -> None:
        """
        Guard fail-closed: verifica que la cuenta es PRACTICE antes de ejecutar.
        Lanza RuntimeError si la verificación falla por cualquier motivo.

        Se activa cuando FORCE_DEMO_ACCOUNT es True (independiente de REMEDIATION_MODE).
        """
        if not FORCE_DEMO_ACCOUNT:
            return

        ts = datetime.now(timezone.utc).isoformat()
        try:
            account_type = self.get_account_type()
        except Exception as e:
            _security_logger.warning(
                f"[{ts}] HALT_TYPE=DEMO_GUARD asset={asset} direction={direction} "
                f"account_type_detected=UNKNOWN force_demo={FORCE_DEMO_ACCOUNT} "
                f"remediation_mode={REMEDIATION_MODE} caller_function={caller} "
                f"stack_trace={traceback.format_exc(limit=3).strip()}"
            )
            raise RuntimeError(
                f"REMEDIATION MODE — Cannot verify account type: {e}. "
                "Trade blocked for safety."
            ) from e

        if account_type not in ALLOWED_ACCOUNT_TYPES:
            _security_logger.warning(
                f"[{ts}] HALT_TYPE=DEMO_GUARD asset={asset} direction={direction} "
                f"account_type_detected={account_type} force_demo={FORCE_DEMO_ACCOUNT} "
                f"remediation_mode={REMEDIATION_MODE} caller_function={caller} "
                f"stack_trace={traceback.format_stack(limit=5)}"
            )
            raise RuntimeError(
                f"REMEDIATION MODE — Account type '{account_type}' is not allowed. "
                f"Only {ALLOWED_ACCOUNT_TYPES} permitted. Trade blocked."
            )

    def buy_binary(
        self, asset: str, amount: float, direction: str, expiry_minutes: int
    ) -> bool:
        """
        Coloca una orden de opción binaria o turbo.

        Args:
            asset:          Nombre del activo (e.g., "EURUSD")
            amount:         Monto en la moneda de la cuenta
            direction:      "call" o "put"
            expiry_minutes: Tiempo de expiración en minutos (1–5)

        Returns:
            (bool, int): Tupla con estado de aceptación y el order_id.
        """
        self._enforce_demo_guard(asset, direction, "buy_binary")

        if not self.is_connected():
            logger.error("buy_binary: sin conexión.")
            return False, -1
        try:
            check, order_id = self.api.buy(amount, asset, direction, expiry_minutes)
            if check:
                logger.info(
                    f"[ORDEN BINARIA ✓] ID:{order_id} | "
                    f"{asset} {direction.upper()} ${amount:.2f} {expiry_minutes}m"
                )
            else:
                logger.error(
                    f"[ORDEN BINARIA ✗] {asset} {direction.upper()} | Razón: {order_id}"
                )
            return bool(check), order_id
        except Exception as e:
            logger.error(f"buy_binary excepción: {e}", exc_info=True)
            return False, -1

    def check_win(self, order_id: int) -> Optional[float]:
        """
        Consulta el resultado de una opción binaria ya expirada.

        Returns:
            float positivo → WIN (ganancia neta)
            float negativo → LOSS (pérdida neta)
            0.0            → empate (tie)
            None           → error o sin conexión
        """
        if not self.is_connected():
            return None
        try:
            result = self.api.check_win_v3(order_id)
            return float(result)
        except Exception as e:
            logger.error(f"check_win: error order {order_id}: {e}")
            return None

    def buy_digital(
        self, asset: str, amount: float, direction: str, expiry_minutes: int
    ) -> bool:
        """
        Coloca una orden de opción digital.
        Las opciones digitales de IQ Option solo aceptan duraciones de 1m o 5m.
        """
        self._enforce_demo_guard(asset, direction, "buy_digital")

        if not self.is_connected():
            logger.error("buy_digital: sin conexión.")
            return False, -1
        # Redondear al valor soportado más cercano
        duration = 5 if expiry_minutes >= 3 else 1
        try:
            check, order_id = self.api.buy_digital_spot(
                asset, amount, direction, duration
            )
            if check:
                logger.info(
                    f"[ORDEN DIGITAL ✓] ID:{order_id} | "
                    f"{asset} {direction.upper()} ${amount:.2f} {duration}m"
                )
            else:
                logger.error(
                    f"[ORDEN DIGITAL ✗] {asset} {direction.upper()} | Razón: {order_id}"
                )
            return bool(check), order_id
        except Exception as e:
            logger.error(f"buy_digital excepción: {e}", exc_info=True)
            return False, -1


# ─── Instancia singleton del servicio ────────────────────────────────────────
iq_service = IQService()
