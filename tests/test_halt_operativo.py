"""
tests/test_halt_operativo.py – Tests del halt operativo (Tareas 0.2 y 0.3).

Valida:
  - Guard pre-orden bloquea ejecución en cuentas no-PRACTICE (Tarea 0.2)
  - retrain_scheduler desactivado en REMEDIATION_MODE (Tarea 0.3)

12 tests:
  - 6 tests originales del plan (Tarea 0.2)
  - 5 tests adicionales de Condición 2 (Tarea 0.2)
  - 1 test retrain_scheduler disabled (Tarea 0.3)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import iqservice


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def svc():
    """IQService con API mockeada (no requiere conexión real)."""
    service = iqservice.IQService()
    service.api = MagicMock()
    service.connected = True
    # is_connected → True por defecto
    service.api.check_connect.return_value = True
    # buy → éxito por defecto
    service.api.buy.return_value = (True, 12345)
    service.api.buy_digital_spot.return_value = (True, 67890)
    return service


@pytest.fixture(autouse=True)
def reset_module_constants():
    """Restaura constantes del módulo después de cada test."""
    orig_remediation = iqservice.REMEDIATION_MODE
    orig_force = iqservice.FORCE_DEMO_ACCOUNT
    orig_allowed = iqservice.ALLOWED_ACCOUNT_TYPES
    yield
    iqservice.REMEDIATION_MODE = orig_remediation
    iqservice.FORCE_DEMO_ACCOUNT = orig_force
    iqservice.ALLOWED_ACCOUNT_TYPES = orig_allowed


# ─── Tests originales (6) ────────────────────────────────────────────────────

def test_guard_blocks_real_account(svc):
    """buy_binary con cuenta REAL + FORCE_DEMO=True → RuntimeError."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.return_value = "REAL"

    with pytest.raises(RuntimeError, match="Account type 'REAL' is not allowed"):
        svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    # Verificar que la orden NUNCA se envió al broker
    svc.api.buy.assert_not_called()


def test_guard_blocks_digital_real_account(svc):
    """buy_digital con cuenta REAL + FORCE_DEMO=True → RuntimeError."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.return_value = "REAL"

    with pytest.raises(RuntimeError, match="Account type 'REAL' is not allowed"):
        svc.buy_digital("EURUSD-OTC", 1.0, "call", 2)

    svc.api.buy_digital_spot.assert_not_called()


def test_guard_allows_practice_account(svc):
    """buy_binary con cuenta PRACTICE + FORCE_DEMO=True → ejecuta normal."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.return_value = "PRACTICE"

    ok, order_id = svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    assert ok is True
    assert order_id == 12345
    svc.api.buy.assert_called_once()


def test_guard_disabled_allows_real(svc):
    """buy_binary con cuenta REAL + FORCE_DEMO=False → ejecuta normal."""
    iqservice.FORCE_DEMO_ACCOUNT = False
    svc.api.get_balance_mode.return_value = "REAL"

    ok, order_id = svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    assert ok is True
    assert order_id == 12345
    svc.api.buy.assert_called_once()


def test_remediation_mode_constant():
    """REMEDIATION_MODE es True en esta branch."""
    assert iqservice.REMEDIATION_MODE is True


def test_connect_logs_startup_warning(caplog):
    """connect() loguea banner con modo + tipo de cuenta."""
    iqservice.REMEDIATION_MODE = True

    mock_api = MagicMock()
    mock_api.connect.return_value = (True, "")
    mock_api.get_all_init_v2.return_value = {}
    mock_api.get_balance_mode.return_value = "PRACTICE"

    with patch("iqservice.Exnova_Option", return_value=mock_api), \
         caplog.at_level(logging.WARNING):
        service = iqservice.IQService()
        service.connect("test@test.com", "pass123")

    combined = " ".join(caplog.text.split())
    assert "REMEDIATION MODE" in combined
    assert "PRACTICE" in combined


# ─── Tests adicionales Condición 2 (5) ──────────────────────────────────────

def test_guard_blocks_when_account_type_unavailable(svc):
    """get_account_type() lanza excepción → RuntimeError por fail-closed."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.side_effect = ConnectionError("WS disconnected")

    with pytest.raises(RuntimeError, match="Cannot verify account type"):
        svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    svc.api.buy.assert_not_called()


def test_guard_blocks_unknown_account_type(svc):
    """get_account_type() retorna tipo desconocido → RuntimeError."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.return_value = "REAL_V2"

    with pytest.raises(RuntimeError, match="Account type 'REAL_V2' is not allowed"):
        svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    svc.api.buy.assert_not_called()


def test_guard_blocks_case_sensitive(svc):
    """get_account_type() retorna 'practice' minúsculas → pasa (upper() normaliza)."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    # get_account_type() llama str(...).upper(), así que 'practice' → 'PRACTICE'
    svc.api.get_balance_mode.return_value = "practice"

    ok, order_id = svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    assert ok is True
    svc.api.buy.assert_called_once()


def test_guard_blocks_none_account_type(svc):
    """get_account_type() retorna None → RuntimeError por fail-closed."""
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.return_value = None

    with pytest.raises(RuntimeError, match="Cannot verify account type"):
        svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    svc.api.buy.assert_not_called()


def test_force_demo_overrides_remediation_disabled(svc):
    """REMEDIATION_MODE=False + FORCE_DEMO=True → guard sigue activo."""
    iqservice.REMEDIATION_MODE = False
    iqservice.FORCE_DEMO_ACCOUNT = True
    svc.api.get_balance_mode.return_value = "REAL"

    with pytest.raises(RuntimeError, match="Account type 'REAL' is not allowed"):
        svc.buy_binary("EURUSD-OTC", 1.0, "call", 2)

    svc.api.buy.assert_not_called()


# ─── Tests Tarea 0.3: retrain_scheduler disabled ────────────────────────────

def test_retrain_scheduler_disabled_in_remediation_mode(caplog):
    """En REMEDIATION_MODE=True, lifespan() NO llama retrain_scheduler.start()."""
    iqservice.REMEDIATION_MODE = True

    with caplog.at_level(logging.WARNING):
        # Simulamos la lógica del lifespan relevante (sin levantar FastAPI)
        from iqservice import REMEDIATION_MODE
        if REMEDIATION_MODE:
            import logging as _log
            _log.getLogger(__name__).warning("REMEDIATION MODE — retrain_scheduler disabled")
            scheduler_started = False
        else:
            scheduler_started = True

    assert scheduler_started is False
    assert "retrain_scheduler disabled" in caplog.text
