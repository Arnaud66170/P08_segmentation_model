# scripts/gpu_check.py

import tensorflow as tf
from tensorflow.python.client import device_lib
import os

def print_div(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

def check_tensorflow_version():
    print_div("🔢 VERSION TENSORFLOW")
    print("TensorFlow version :", tf.__version__)

def list_available_devices():
    print_div("🔍 DISPOSITIFS RECONNUS")
    devices = device_lib.list_local_devices()
    for d in devices:
        print(f"{d.device_type} | {d.name} | Memory: {getattr(d, 'memory_limit', 'N/A')}")

def check_gpu_available():
    print_div("✅ VERIFICATION GPU")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✔️  {len(gpus)} GPU détecté(s) :")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("❌ Aucun GPU détecté. L'entraînement se fera sur CPU.")

def activate_memory_growth():
    print_div("⚙️ PARAMÉTRAGE DE LA MÉMOIRE")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⏭️ Skip : Aucun GPU à configurer.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✔️  Mémoire GPU configurée en mode 'growth'.")
    except RuntimeError as e:
        print("❌ Erreur lors de la configuration de la mémoire :", str(e))

def print_cuda_env():
    print_div("🧪 ENV CUDA / cuDNN")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "Non défini"))
    print("TF CUDA Built with CUDA :", tf.test.is_built_with_cuda())
    print("TF CUDA Built with GPU support :", tf.test.is_built_with_gpu_support())
    print("TF CUDA Compatible version :", tf.sysconfig.get_build_info().get('cuda_version', 'inconnu'))
    print("TF cuDNN version :", tf.sysconfig.get_build_info().get('cudnn_version', 'inconnu'))

if __name__ == "__main__":
    check_tensorflow_version()
    check_gpu_available()
    activate_memory_growth()
    list_available_devices()
    print_cuda_env()
