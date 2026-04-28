from iqservice import Exnova_Option
import time

# --- CONFIGURACIÓN DE PRUEBA ---
USER = "felipemartinezleiva96@gmail.com"
PASS = "Nuttertools2026***"
MONTO = 1000
ACCION = "put"        # "call" para sube, "put" para baja
TEMPORALIDAD = 3       # 3 minutos
ACTIVO = "AUDCAD-OTC"      # Activo específico a probar

print("Iniciando prueba de ejecución forzada en Exnova...")
api = Exnova_Option(USER, PASS)
check, reason = api.connect()

if check:
    api.change_balance("PRACTICE")
    print(f"Conectado. Probando entrada en {ACTIVO} a {TEMPORALIDAD} minutos...")
    
    try:
        status, orden_id = api.buy(MONTO, ACTIVO, ACCION, TEMPORALIDAD)
        
        if status:
            print(f"✅ ¡ÉXITO! Operación abierta a {TEMPORALIDAD}m con ID: {orden_id} en {ACTIVO}")
            print("Tu conexión y ejecución funcionan perfectamente.")
        else:
            print(f"❌ Falló la orden. El broker dijo: {orden_id}")
    except Exception as e:
        print(f"❌ Error en la llamada API: {e}")
else:
    print(f"❌ Error de conexión: {reason}")