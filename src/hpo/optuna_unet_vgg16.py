# src/hpo/optuna_unet_vgg16.py

import optuna
import mlflow
import mlflow.keras
import numpy as np
import joblib
from pathlib import Path
from model_training.train_unet_vgg16 import train_unet_vgg16

# 📁 Chemins
project_root = Path("..").resolve()
processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"

# Chargement des données
train_path = processed_dir / "train.npz"
val_path = processed_dir / "val.npz"

data_train = np.load(train_path)
data_val = np.load(val_path)

X_train, y_train = data_train["X"], data_train["Y"]
X_val, y_val     = data_val["X"], data_val["Y"]

# 🎯 Fonction objectif pour Optuna
def objective(trial):
    # 🔧 Espace de recherche
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    epochs = trial.suggest_int("epochs", 20, 50, step=10)
    loss_function = trial.suggest_categorical("loss_function", ["sparse_categorical_crossentropy"])
    model_name = f"optuna_vgg16_bs{batch_size}_ep{epochs}"

    # 🚀 Lancement du training
    with mlflow.start_run(nested=True):
        model, history = train_unet_vgg16(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=str(models_dir),
            model_name=model_name,
            force_retrain=True,
            epochs=epochs,
            batch_size=batch_size,
            loss_function=loss_function,
            use_early_stopping=True
        )

        val_accuracy = max(history.history["val_accuracy"])
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "loss_function": loss_function
        })
        mlflow.log_metric("best_val_accuracy", val_accuracy)

        return 1.0 - val_accuracy  # Minimisation

# 🚀 Fonction de lancement de l'étude

def launch_optuna_study(n_trials=10):
    study = optuna.create_study(direction="minimize", study_name="unet_vgg16_hpo")
    study.optimize(objective, n_trials=n_trials)

    print("\n🏆 Meilleurs hyperparamètres :")
    print(study.best_trial.params)

    # Sauvegarde de l'étude
    study_path = models_dir / "optuna_vgg16_study.pkl"
    joblib.dump(study, study_path)
    print(f"\n💾 Étude sauvegardée dans : {study_path}")

    return study