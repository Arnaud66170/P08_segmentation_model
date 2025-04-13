# src/data_preprocessing/preprocessing.py
# Preprocessing pipeline pour Cityscapes (dossier plat généré via extract_cityscapes_flat.py)

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mlflow
from .class_mapping import CLASS_MAPPING_P8

from utils.logger import log_step


@log_step
def resize_image(image, size=(256, 256)):
    """Redimensionne une image (ou mask) à une taille fixe"""
    return cv2.resize(image, size, interpolation = cv2.INTER_NEAREST)

@log_step
def normalize_image(image):
    """Normalisation simple des pixels RGB entre 0 et 1"""
    return image / 255.0

@log_step
def map_mask_to_8_classes(mask, mapping_dict):
    """Remapping des pixels mask depuis les classes Cityscapes → vers 8 classes principales"""
    result = np.full_like(mask, fill_value = 255)
    for orig_id, new_id in mapping_dict.items():
        result[mask == orig_id] = new_id
    return result

@log_step
def prepare_dataset(image_dir, mask_dir, output_dir, mapping_dict = CLASS_MAPPING_P8, img_size = (256, 256)):
    """
    Étape clé du pipeline :
    - lit les images et masks du dossier plat
    - resize et normalise les images
    - remap les masks avec 8 classes
    - split en train / val / test
    - sauvegarde en fichiers compressés .npz
    - loggue les artefacts et paramètres avec MLflow
    """

    images = sorted(glob(os.path.join(image_dir, "*.png")))
    masks = sorted(glob(os.path.join(mask_dir, "*.png")))

    print(f"[DEBUG] {len(images)} images trouvées")
    print(f"[DEBUG] {len(masks)} masks trouvés")

    if not images or not masks:
        raise ValueError("Aucune image ou aucun mask trouvé dans les dossiers fournis.")

    if len(images) != len(masks):
        raise ValueError(f"Incohérence : {len(images)} images vs {len(masks)} masks. Les fichiers ne sont pas appariés.")

    X, Y = [], []
    for img_path, mask_path in tqdm(zip(images, masks), total=len(images)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_resized = resize_image(img, img_size)
        img_norm = normalize_image(img_resized)
        mask_resized = resize_image(mask, img_size)
        mask_mapped = map_mask_to_8_classes(mask_resized, mapping_dict)

        X.append(img_norm)
        Y.append(mask_mapped)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.uint8)

    # Split des données
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "train.npz"), X=X_train, Y=y_train)
    np.savez_compressed(os.path.join(output_dir, "val.npz"), X=X_val, Y=y_val)
    np.savez_compressed(os.path.join(output_dir, "test.npz"), X=X_test, Y=y_test)

    with mlflow.start_run(run_name = "preprocessing_pipeline"):
        mlflow.log_param("image_size", img_size)
        mlflow.log_param("total_images", len(images))
        mlflow.log_param("train_split", len(X_train))
        mlflow.log_param("val_split", len(X_val))
        mlflow.log_param("test_split", len(X_test))

        mlflow.log_artifact(os.path.join(output_dir, "train.npz"))
        mlflow.log_artifact(os.path.join(output_dir, "val.npz"))
        mlflow.log_artifact(os.path.join(output_dir, "test.npz"))

    print("[INFO] Données sauvegardées et loggées avec MLflow dans :", output_dir)

@log_step
def load_data_npz(path: str):
    """
    Charge les arrays de données prétraités depuis un fichier .npz.
    Retourne : X_train, y_train, X_val, y_val
    """
    with np.load(path) as data:
        return data["X_train"], data["y_train"], data["X_val"], data["y_val"]