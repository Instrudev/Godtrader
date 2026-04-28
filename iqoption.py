from iqoptionapi.stable_api import IQ_Option
import getpass

# Solicitar credenciales al usuario de forma segura
email = input("Ingresa tu email: ")
password = getpass.getpass("Ingresa tu contraseña: ")

# Inicializar la API
Iq = IQ_Option(email, password)

# Intentar la conexión
check, reason = Iq.connect()

if check:
    print("¡Conexión exitosa!")
else:
    print(f"Error al conectar: {reason}")

# Opcional: Cambiar a cuenta de práctica (PRACTICE) o real (REAL)
# Iq.change_balance('PRACTICE')