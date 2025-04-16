# src/evaluation/evaluation.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import jaccard_score, f1_score
import mlflow

# Calcul du IoU et du Dice coefficient pour un modèle donné
# (et log des métriques dans MLflow pour traçabilité MLOps)
def evaluate_model(model_path, model_name, X_val, y_val):
    print(f"\n🔍 Évaluation du modèle : {model_name}")
    try:
        model = load_model(model_path, compile=False)
        y_pred = model.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=-1).flatten()
        y_true_labels = y_val.flatten()

        iou = jaccard_score(y_true_labels, y_pred_labels, average='macro')
        dice = f1_score(y_true_labels, y_pred_labels, average='macro')

        # 🔁 Log MLOps
        with mlflow.start_run(run_name=f"eval_{model_name}", nested=True):
            mlflow.log_param("eval_model", model_name)
            mlflow.log_metric("IoU", iou)
            mlflow.log_metric("Dice", dice)

        return {
            "model": model_name,
            "IoU": iou,
            "Dice": dice
        }
    except Exception as e:
        print(f"[ERREUR] Échec de l'évaluation pour {model_name} : {e}")
        return None


# Evaluation d'un modèle déjà chargé en mémoire
# (utile pour éviter de recharger le modèle à chaque fois)
def evaluate_loaded_model(model, model_name, X_val, y_val):
    print(f"\n🔍 Évaluation du modèle en mémoire : {model_name}")
    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=-1).flatten()
    y_true_labels = y_val.flatten()

    iou = jaccard_score(y_true_labels, y_pred_labels, average='macro')
    dice = f1_score(y_true_labels, y_pred_labels, average='macro')

    print(f"   ➤ IoU : {iou:.4f}")
    print(f"   ➤ Dice coefficient : {dice:.4f}")

    with mlflow.start_run(run_name=f"eval_{model_name}_loaded", nested=True):
        mlflow.log_param("eval_model", model_name)
        mlflow.log_metric("IoU", iou)
        mlflow.log_metric("Dice", dice)

    return {
        "IoU": iou,
        "Dice": dice
    }