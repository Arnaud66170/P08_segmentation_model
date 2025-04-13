import subprocess
import socket
import os
import time
import mlflow
from functools import wraps

MLFLOW_PORT = 5000
MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"

def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_mlflow_server_if_needed():
    if not is_port_open(MLFLOW_PORT):
        print("🔄 Lancement du serveur MLflow local...")
        subprocess.Popen(
            ["mlflow", "server", "--backend-store-uri", "mlruns", "--port", str(MLFLOW_PORT)],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL
        )
        time.sleep(5)
        print(f"✅ Serveur MLflow démarré sur {MLFLOW_TRACKING_URI}")
    else:
        print(f"✅ Serveur MLflow déjà actif sur {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def mlflow_logging_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_mlflow_server_if_needed()
        return func(*args, **kwargs)
    return wrapper
