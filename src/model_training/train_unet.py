# src/model_training/train_unet.py

import os
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from datetime import datetime
import joblib
import pandas as pd
import yaml
import time
import GPUtil
import psutil

from data_generator.generator import AlbumentationDataGenerator
from model_training.metrics import iou_score, dice_coef
from utils.mlflow_manager import mlflow_logging_decorator
from utils.utils import plot_history
from utils.logger import log_step

ARTIFACTS_DIR = Path("models")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Chargement config GPU/batch (chemin absolu)
config_path = Path(__file__).resolve().parents[2] / "config" / "config_gpu.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
BATCH_SIZE = config['gpu']['batch_size']

# Détection GPU et configuration mémoire dynamique
print("[INFO] Configuration GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU détecté : {gpus}")
    except RuntimeError as e:
        print(f"❌ Erreur de config GPU : {e}")
else:
    print("⚠️ Aucun GPU détecté. Utilisation du CPU.")

# Visualisation GPU (optionnel)
try:
    GPUtil.showUtilization()
except:
    print("⚠️ GPUtil non disponible")

# Fonction de monitoring CPU + GPU
def monitor_resources_live(duration=30, interval=3):
    import shutil
    for _ in range(duration // interval):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        gpus = GPUtil.getGPUs()
        gpu = gpus[0] if gpus else None

        os.system('cls' if os.name == 'nt' else 'clear')
        print("🔁 Surveillance CPU/GPU (LIVE - rafraichi toutes les 3s)")
        print("=" * 60)
        print(f"🧠 CPU Usage : {cpu}%")
        print(f"🧠 RAM Usage : {ram.percent}% ({round(ram.used/1e9, 1)}GB / {round(ram.total/1e9, 1)}GB)")
        if gpu:
            print(f"🎮 GPU: {gpu.name}")
            print(f"   Utilisation : {gpu.load*100:.1f}%")
            print(f"   RAM : {gpu.memoryUsed:.0f} / {gpu.memoryTotal:.0f} MB")
        print("=" * 60)
        time.sleep(interval)

@mlflow_logging_decorator
@log_step
def unet_mini(input_shape=(256, 256, 3), num_classes=8):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c4)
    return keras.Model(inputs, outputs)


@mlflow_logging_decorator
@log_step
def train_unet_model_from_npz(X_train, Y_train, X_val, Y_val,
                              force_retrain=False,
                              img_size=(256, 256),
                              epochs=20,
                              batch_size=BATCH_SIZE,
                              use_early_stopping=True,
                              num_classes=8,
                              turbo=False):

    if turbo:
        print("🚀 Mode TURBO activé : optimisations en cours...")
        tf.config.optimizer.set_jit(True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        batch_size = 8

    model_name = f"unet_mini_npz_{img_size[0]}x{img_size[1]}_bs{batch_size}_ep{epochs}"
    if turbo:
        model_name += "_TURBO"

    model_path   = ARTIFACTS_DIR / f"{model_name}.h5"
    history_path = ARTIFACTS_DIR / f"{model_name}_history.pkl"
    plot_path    = ARTIFACTS_DIR / f"{model_name}_training_plot.png"
    csv_path     = ARTIFACTS_DIR / f"{model_name}_history.csv"

    if model_path.exists() and not force_retrain:
        print(f"[INFO] Modèle déjà existant : {model_path}")
        model = keras.models.load_model(
            model_path,
            custom_objects={"iou_score": iou_score, "dice_coef": dice_coef}
        )
        history = joblib.load(history_path)
        return model, history

    print("[INFO] Initialisation du modèle...")
    model = unet_mini(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', iou_score, dice_coef]
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
    if not turbo:
        callbacks.append(tf.keras.callbacks.CSVLogger(str(csv_path)))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), save_best_only=True))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "input_shape": img_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": use_early_stopping,
            "force_retrain": force_retrain,
            "augmentations": "None (npz direct)",
            "turbo_mode": turbo
        })

        print("[INFO] Début entraînement depuis npz...")

        if turbo:
            import threading
            t = threading.Thread(target=monitor_resources_live, args=(epochs * 5,))
            t.start()
        else:
            monitor_resources()  # affichage ponctuel pour les modes classiques

        start = time.time()

        history_obj = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose = 1  # ← tu peux remettre à 1 pour afficher la progression
        )
        end = time.time()

        duration = round(end - start, 2)
        print(f"[INFO] Temps total entraînement : {duration} secondes")
        mlflow.log_metric("train_time_sec", duration)

        joblib.dump(history_obj.history, history_path)
        pd.DataFrame(history_obj.history).to_csv(csv_path, index=False)
        plot_history(history_obj, plot_path)

        mlflow.keras.log_model(model, model_name)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(csv_path))

        for epoch in range(len(history_obj.history['loss'])):
            mlflow.log_metric("loss", history_obj.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history_obj.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("accuracy", history_obj.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history_obj.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("iou_score", history_obj.history['iou_score'][epoch], step=epoch)
            mlflow.log_metric("val_iou_score", history_obj.history['val_iou_score'][epoch], step=epoch)
            mlflow.log_metric("dice_coef", history_obj.history['dice_coef'][epoch], step=epoch)
            mlflow.log_metric("val_dice_coef", history_obj.history['val_dice_coef'][epoch], step=epoch)

        run = mlflow.active_run()
        print(f"📊 MLflow Run enregistré : http://127.0.0.1:5000/#/experiments/0/runs/{run.info.run_id}")

    return model, history_obj.history

# Nouvelle fonction avec Albumentation + mode turbo + chemins défaut
@mlflow_logging_decorator
@log_step
def train_unet_model_albumentation(train_dir=None,
                                   val_dir=None,
                                    force_retrain=False,
                                    img_size=(256, 256),
                                    epochs=20,
                                    batch_size=BATCH_SIZE,
                                    use_early_stopping=True,
                                    num_classes=8,
                                    turbo=True):
    if train_dir is None or val_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        train_dir = project_root / "data" / "processed" / "train"
        val_dir = project_root / "data" / "processed" / "val"
        
    if turbo:
        print("🚀 Mode TURBO activé : optimisations en cours...")
        tf.config.optimizer.set_jit(True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        batch_size = 8

    model_name = f"unet_mini_albu_{img_size[0]}x{img_size[1]}_bs{batch_size}_ep{epochs}"
    if turbo:
        model_name += "_TURBO"

    model_path   = ARTIFACTS_DIR / f"{model_name}.h5"
    history_path = ARTIFACTS_DIR / f"{model_name}_history.pkl"
    plot_path    = ARTIFACTS_DIR / f"{model_name}_training_plot.png"
    csv_path     = ARTIFACTS_DIR / f"{model_name}_history.csv"

    if model_path.exists() and not force_retrain:
        print(f"[INFO] Modèle déjà existant : {model_path}")
        model = keras.models.load_model(
            model_path,
            custom_objects={"iou_score": iou_score, "dice_coef": dice_coef}
        )
        history = joblib.load(history_path)
        return model, history

    print("[INFO] Initialisation des DataGenerators...")
    train_gen = AlbumentationDataGenerator(
        image_dir=train_dir / "images",
        mask_dir=train_dir / "masks",
        batch_size=batch_size,
        img_size=img_size,
        augment=True
    )
    val_gen = AlbumentationDataGenerator(
        image_dir=val_dir / "images",
        mask_dir=val_dir / "masks",
        batch_size=batch_size,
        img_size=img_size,
        augment=False
    )

    print("[INFO] Initialisation du modèle...")
    model = unet_mini(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', iou_score, dice_coef]
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
    if not turbo:
        callbacks.append(tf.keras.callbacks.CSVLogger(str(csv_path)))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), save_best_only=True))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "input_shape": img_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": use_early_stopping,
            "force_retrain": force_retrain,
            "augmentations": "Albumentation",
            "turbo_mode": turbo
        })

        print("[INFO] Début entraînement avec Albumentation...")

        if turbo:
            import threading
            t = threading.Thread(target=monitor_resources_live, args=(epochs * 5,))
            t.start()
        else:
            monitor_resources_live(duration=15)

        start = time.time()
        history_obj = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            workers=0,
            use_multiprocessing=False,
            verbose=1
        )
        end = time.time()

        duration = round(end - start, 2)
        print(f"[INFO] Temps total entraînement : {duration} secondes")
        mlflow.log_metric("train_time_sec", duration)

        joblib.dump(history_obj.history, history_path)
        pd.DataFrame(history_obj.history).to_csv(csv_path, index=False)
        plot_history(history_obj, plot_path)

        mlflow.keras.log_model(model, model_name)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(csv_path))

        for epoch in range(len(history_obj.history['loss'])):
            mlflow.log_metric("loss", history_obj.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history_obj.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("accuracy", history_obj.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history_obj.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("iou_score", history_obj.history['iou_score'][epoch], step=epoch)
            mlflow.log_metric("val_iou_score", history_obj.history['val_iou_score'][epoch], step=epoch)
            mlflow.log_metric("dice_coef", history_obj.history['dice_coef'][epoch], step=epoch)
            mlflow.log_metric("val_dice_coef", history_obj.history['val_dice_coef'][epoch], step=epoch)

        run = mlflow.active_run()
        print(f"📊 MLflow Run enregistré : http://127.0.0.1:5000/#/experiments/0/runs/{run.info.run_id}")

    return model, history_obj.history

def fetch_mlflow_runs(experiment_name="Default"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(experiment.experiment_id)
    rows = []
    for run in runs:
        data = run.data
        row = {
            "Run ID": run.info.run_id,
            "Model Name": data.tags.get("mlflow.runName", ""),
            "Turbo": data.params.get("turbo_mode", "False"),
            "Batch Size": data.params.get("batch_size"),
            "Epochs": data.params.get("epochs"),
            "Final val_accuracy": data.metrics.get("val_accuracy"),
            "Final val_loss": data.metrics.get("val_loss"),
            "Train Time (s)": data.metrics.get("train_time_sec"),
        }
        rows.append(row)
    return pd.DataFrame(rows)