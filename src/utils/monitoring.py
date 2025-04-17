# zrc/utils/monitoring.py

import psutil
import GPUtil
import mlflow

def monitor_resources():
    """
    Affiche un résumé de l’utilisation des ressources (RAM, CPU, GPU).
    Utile en mode TURBO ou pour debug de performance.
    """
    print("\n🔍 MONITORING DES RESSOURCES :")

    # RAM
    ram = psutil.virtual_memory()
    print(f"🧠 RAM utilisée : {ram.percent}% ({round(ram.used / 1024**3, 2)} Go / {round(ram.total / 1024**3, 2)} Go)")
    print(f"   RAM disponible : {round(ram.available / 1024**3, 2)} Go")

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"🖥️  CPU utilisé : {cpu_percent}%")

    # GPU
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("🚫 Aucun GPU détecté par GPUtil.")
        for gpu in gpus:
            print(f"\n🚀 GPU détecté : {gpu.name}")
            print(f"   ➤ Charge : {gpu.load * 100:.1f}%")
            print(f"   ➤ Mémoire : {gpu.memoryUsed} / {gpu.memoryTotal} Mo ({gpu.memoryUtil * 100:.1f}%)")
    except Exception as e:
        print(f"❌ Erreur GPUtil : {e}")


def log_resources_to_mlflow():
    """
    Log l’état CPU/RAM/GPU actuel dans MLflow.
    """
    ram = psutil.virtual_memory()
    mlflow.log_metric("RAM_used_percent", ram.percent)
    mlflow.log_metric("RAM_available_GB", round(ram.available / 1024**3, 2))
    mlflow.log_metric("CPU_usage_percent", psutil.cpu_percent(interval=1))

    try:
        gpu = GPUtil.getGPUs()[0]
        mlflow.log_metric("GPU_load_percent", gpu.load * 100)
        mlflow.log_metric("GPU_mem_used_MB", gpu.memoryUsed)
    except:
        pass