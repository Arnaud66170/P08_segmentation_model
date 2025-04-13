# src/model_training/train_unet.py (✅ MLflow-ready + courbes interactives + logs)

import os
import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from datetime import datetime
import joblib

# 📦 Custom generator (assume src structure)
from data_generator.generator import CityscapesDataGenerator
from utils.mlflow_manager import mlflow_logging_decorator
from utils.utils import masked_sparse_categorical_crossentropy, plot_history
from utils.logger import log_step

# 📁 Où sauvegarder les modèles, métriques, etc.
ARTIFACTS_DIR = Path("models")
ARTIFACTS_DIR.mkdir(exist_ok=True)

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
def train_unet_model(X_train, Y_train, X_val, Y_val, 
                     force_retrain=False, 
                     img_size=(256, 256), 
                     epochs=20, 
                     batch_size=8, 
                     use_early_stopping=True):

    model_name = f"unet_mini_cityscapes_{img_size[0]}x{img_size[1]}"
    model_path = ARTIFACTS_DIR / f"{model_name}.h5"
    history_path = ARTIFACTS_DIR / f"{model_name}_history.pkl"
    plot_path = ARTIFACTS_DIR / f"{model_name}_training_plot.png"
    csv_path = ARTIFACTS_DIR / f"{model_name}_history.csv"

    if model_path.exists() and not force_retrain:
        print("[INFO] Modèle déjà existant. Chargement...")
        model = keras.models.load_model(
            model_path,
            custom_objects={"masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy}
        )
        history = joblib.load(history_path)
        return model, history

    print("[INFO] Entraînement du modèle U-Net...")
    train_gen = CityscapesDataGenerator(X_train, Y_train, augment=True, batch_size=batch_size)
    val_gen   = CityscapesDataGenerator(X_val, Y_val, augment=False, batch_size=batch_size)

    model = unet_mini(input_shape=(img_size[0], img_size[1], 3))
    model.compile(
        optimizer='adam',
        loss=masked_sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    callbacks = []
    if use_early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        callbacks.append(early_stop)

    with mlflow.start_run(run_name=f"unet_bs{batch_size}_ep{epochs}_img{img_size[0]}"):
        mlflow.log_params({
            "input_shape": img_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": use_early_stopping,
            "force_retrain": force_retrain
        })

        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

        model.save(model_path)
        joblib.dump(history.history, history_path)

        import pandas as pd
        pd.DataFrame(history.history).to_csv(csv_path, index=False)

        plot_history(history, plot_path)

        mlflow.keras.log_model(model, model_name)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(csv_path))

        # Logging explicite des métriques par epoch (visibles en graphe dans MLflow UI)
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

    return model, history