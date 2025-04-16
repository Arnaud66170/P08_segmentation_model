# src/model_training/train_unet_vgg16.py (corrigé, enrichi, compatible notebooks + MLflow + visualisation dynamique)

import os
import time
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.keras

from utils.logger import log_step
from utils.mlflow_manager import mlflow_logging_decorator
from utils.utils import plot_history

@mlflow_logging_decorator
@log_step
def train_unet_vgg16(X_train, y_train, X_val, y_val, 
                     output_dir = "models/", 
                     model_name = "unet_vgg16", 
                     force_retrain = False, 
                     epochs = 30, 
                     batch_size = 16, 
                     loss_function = "sparse_categorical_crossentropy",
                     use_early_stopping = True,
                     test_mode = False):
    
    os.makedirs(output_dir, exist_ok = True)

    model_path   = os.path.join(output_dir, f"{model_name}.h5")
    history_path = os.path.join(output_dir, f"{model_name}_history.pkl")
    plot_path    = os.path.join(output_dir, f"{model_name}_training_plot.png")
    metrics_csv  = os.path.join(output_dir, f"{model_name}_metrics.csv")

    if os.path.exists(model_path) and os.path.exists(history_path) and not force_retrain:
        print(f"[INFO] ⟳ Chargement du modèle existant : {model_path}")
        try:
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects={},
                compile=False
            )
            history = joblib.load(history_path)
            return model, history
        except Exception as e:
            print(f"[ERREUR] Échec du chargement du modèle : {e}")
            print("[INFO] Réentraînement forcé...")

    print("[INFO] Entraînement d'un modèle UNet + VGG16...")
    vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=X_train.shape[1:])
    for layer in vgg16.layers:
        layer.trainable = False

    inputs = vgg16.input
    x = vgg16.output

    x = UpSampling2D()(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    num_classes = len(np.unique(y_train))
    outputs = Conv2D(num_classes, 1, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

    if test_mode:
        print("[TEST MODE] ➤ Exécution rapide sur 2 échantillons pour vérif structure/model")
        try:
            model.fit(
                X_train[:2], y_train[:2],
                validation_data=(X_val[:2], y_val[:2]),
                epochs=1,
                batch_size=1,
                verbose=2
            )
            print("[TEST MODE] ✅ OK - entraînement minimal passé sans erreur de shape")
        except Exception as e:
            print("[TEST MODE] ❌ ERREUR détectée lors de l'entraînement de test :")
            raise e
        return model, {"test_mode": True}

    # Callbacks dynamiques
    callbacks = [ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", verbose=1)]
    if use_early_stopping:
        early_stop = EarlyStopping(patience = 5, restore_best_weights = True)
        callbacks.append(early_stop)

    run_id = f"{model_name}_bs{batch_size}_ep{epochs}"
    start = time.time()

    with mlflow.start_run(run_name=run_id, nested=True):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "loss_function": loss_function,
            "model_name": model_name,
            "force_retrain": force_retrain,
            "use_early_stopping": use_early_stopping
        })

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        model.save(model_path)
        joblib.dump(history.history, history_path)
        plot_history(history, plot_path)

        # Log images et objets
        mlflow.keras.log_model(model, model_name)
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(history_path)
        mlflow.log_artifact(model_path)

        # Log CSV des métriques
        import pandas as pd
        pd.DataFrame(history.history).to_csv(metrics_csv, index=False)
        mlflow.log_artifact(metrics_csv)

        # Log dynamique epoch par epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

        duration = round(time.time() - start, 2)
        mlflow.log_metric("training_time_seconds", duration)

    return model, history
