# src/data_preprocessing/preprocessing.py
# Preprocessing pipeline pour Cityscapes (dossier plat généré via extract_cityscapes_flat.py)

# standard
import os
import csv
from pathlib import Path
from datetime import datetime

# Data Science
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# Evalusation / split
from sklearn.model_selection import train_test_split

#MLOps
import mlflow
import matplotlib.pyplot as plt

# Custom (décorateur, logger)
from utils.logger import log_step

# Imports internes
from .class_mapping import FLAT_CLASS_MAPPING


@log_step
def resize_image(image, size=(256, 256)):
    """Redimensionne une image (ou mask) à une taille fixe"""
    return cv2.resize(image, size, interpolation = cv2.INTER_NEAREST)

@log_step
def normalize_image(image):
    """Normalisation simple des pixels RGB entre 0 et 1"""
    return image / 255.0

# @log_step
# def map_mask_to_8_classes(mask, mapping_dict):
#     """Remapping des pixels mask depuis les classes Cityscapes → vers 8 classes principales"""
#     result = np.full_like(mask, fill_value = 255)
#     for orig_id, new_id in mapping_dict.items():
#         result[mask == orig_id] = new_id
#     return result


@log_step
def prepare_dataset(
    image_dir,
    mask_dir,
    output_dir,
    mapping_dict=FLAT_CLASS_MAPPING,
    img_size=(256, 256),
    force_preprocessing=False,
    mlflow_tracking=True
):
    """
    Prépare les données de segmentation :
    - Resize, normalisation des images
    - Mapping des classes (vers superclasses)
    - Split train/val/test
    - Sauvegarde .npz
    - Tracking MLflow + log CSV + figure des distributions
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_exist = all([
        (output_dir / f"{split}.npz").exists()
        for split in ["train", "val", "test"]
    ])

    if paths_exist and not force_preprocessing:
        print(f"[INFO] Données déjà prétraitées présentes dans {output_dir}.")
        return

    if force_preprocessing and paths_exist:
        for f in output_dir.glob("*.npz"):
            f.unlink()
        print(f"[INFO] Fichiers .npz supprimés pour preprocessing forcé.")

    # Chargement des fichiers
    images = sorted(glob(os.path.join(image_dir, "*.png")))
    masks = sorted(glob(os.path.join(mask_dir, "*.png")))

    if not images or not masks:
        raise FileNotFoundError("Aucune image ou mask trouvé dans les dossiers fournis.")
    if len(images) != len(masks):
        raise ValueError(f"Incohérence : {len(images)} images vs {len(masks)} masks.")

    X, Y = [], []
    for img_path, mask_path in tqdm(zip(images, masks), total=len(images)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
        img_norm = img_resized / 255.0
        mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

        remapped_mask = np.full_like(mask_resized, fill_value = 255)
        for src_id, class_id in mapping_dict.items():
            remapped_mask[mask_resized == src_id] = class_id

        X.append(img_norm)
        Y.append(remapped_mask)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.uint8)

    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    np.savez_compressed(output_dir / "train.npz", X=X_train, Y=y_train)
    np.savez_compressed(output_dir / "val.npz", X=X_val, Y=y_val)
    np.savez_compressed(output_dir / "test.npz", X=X_test, Y=y_test)

    if mlflow_tracking:
        with mlflow.start_run(run_name="preprocessing_pipeline"):
            mlflow.log_param("image_size", img_size)
            mlflow.log_param("mapping_classes", len(set(mapping_dict.values())))
            mlflow.log_param("total_images", len(images))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_artifact(str(output_dir / "train.npz"))
            mlflow.log_artifact(str(output_dir / "val.npz"))
            mlflow.log_artifact(str(output_dir / "test.npz"))

    # 📄 Log CSV local
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocessing_log.csv"

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": output_dir.name,
        "img_size": f"{img_size[0]}x{img_size[1]}",
        "mapping_classes": len(set(mapping_dict.values())),
        "total_images": len(images),
        "train": len(X_train),
        "val": len(X_val),
        "test": len(X_test)
    }

    header = list(log_data.keys())
    row = list(log_data.values())
    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[LOG] Infos de preprocessing ajoutées à : {log_file}")

    # 📊 Génération et sauvegarde du graphe de distribution des classes
    def compute_class_distribution(y_sets, nb_classes):
        counts = np.zeros(nb_classes, dtype=int)
        for y in y_sets:
            flat = y.flatten()
            valid = flat[flat != 255]
            counts += np.bincount(valid, minlength=nb_classes)
        return counts

    nb_classes = len(set(mapping_dict.values()))
    global_dist = compute_class_distribution([y_train, y_val, y_test], nb_classes)

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(range(nb_classes), global_dist, color='cornflowerblue')
    plt.title(f"Distribution des classes - Dataset: {output_dir.name}")
    plt.xlabel("Classe ID")
    plt.ylabel("Pixels")
    plt.xticks(range(nb_classes))
    plt.grid(True)
    plt.tight_layout()

    plot_path = figures_dir / f"class_distribution_{output_dir.name}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[FIG] Graphe de distribution sauvegardé dans : {plot_path}")
    print(f"[✅] Preprocessing terminé. Données sauvegardées dans : {output_dir}")

@log_step
def load_data_npz(path: str):
    """
    Charge les arrays de données prétraités depuis un fichier .npz.
    Retourne : X_train, y_train, X_val, y_val
    """
    with np.load(path) as data:
        return data["X_train"], data["y_train"], data["X_val"], data["y_val"]