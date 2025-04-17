# scripts/regiter_kernel.py
# Lancement du noyau pour utilisation GPU
import os
import sys
import subprocess

def main():
    print("🔁 Enregistrement du kernel Jupyter pour tf_gpu_env...")

    # Vérifie que ipykernel est installé
    try:
        import ipykernel
    except ImportError:
        print("⚠️ ipykernel n'est pas installé. Installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])

    # Enregistrement du kernel
    cmd = [
        sys.executable,
        "-m", "ipykernel", "install",
        "--user",
        "--name", "tf_gpu_env",
        "--display-name", "Python (tf_gpu_env)"
    ]

    subprocess.run(cmd, check=True)
    print("✅ Kernel enregistré avec succès !")

if __name__ == "__main__":
    main()

# lancement du script via terminal
# python scripts/register_kernel.py
