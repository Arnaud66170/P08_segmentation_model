# src/data_preprocessing/augmentation.py

import os
from pathlib import Path
import Augmentor
from tqdm import tqdm
from utils.logger import log_step
import mlflow


@log_step
def generate_augmented_dataset(
    image_dir,
    mask_dir,
    output_dir,
    nb_samples=50000,
    img_size=(256, 256),
    force_generate=False,
    mlflow_tracking=True
):
    """
    Génère un dataset augmenté à partir d'un jeu d’images/masks existant.

    - Applique des transformations synchronisées (rotation, flip, zoom, skew, etc.)
    - Redimensionne à img_size
    - Sauvegarde les images augmentées dans output_dir/images & output_dir/masks
    - Peut être relancé en mode 'force' pour écraser les données existantes
    - Traçabilité avec MLflow activable

    Args:
        image_dir (str or Path): Chemin vers les images originales
        mask_dir (str or Path): Chemin vers les masks associés
        output_dir (str or Path): Dossier de sortie contenant les données augmentées
        nb_samples (int): Nombre d’échantillons à générer
        img_size (tuple): Dimension cible pour resize (width, height)
        force_generate (bool): Forcer la régénération même si les fichiers existent
        mlflow_tracking (bool): Active ou non le tracking MLflow
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_images = output_dir / "images"
    output_masks = output_dir / "masks"

    if output_images.exists() and output_masks.exists() and not force_generate:
        print(f"[INFO] Données augmentées déjà présentes dans {output_dir}. Utilise force_generate=True pour forcer la régénération.")
        return

    # Suppression précédente si force
    if force_generate and output_dir.exists():
        print(f"[INFO] Suppression de {output_dir} pour regénération...")
        for sub in [output_images, output_masks]:
            if sub.exists():
                for file in sub.glob("*"):
                    file.unlink()
    
    print(f"[INFO] Initialisation du pipeline d’augmentation sur {image_dir} ...")
    p = Augmentor.Pipeline(str(image_dir), output_directory=str(output_images))
    p.ground_truth(str(mask_dir))

    # 🌪️ Transformations diverses synchronisées
    p.rotate(probability=1.0, max_left_rotation=10, max_right_rotation=10)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.skew(probability=0.5, magnitude=0.5)
    p.skew_tilt(probability=0.5, magnitude=0.5)
    p.random_distortion(probability=0.5, grid_height=4, grid_width=4, magnitude=4)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.gaussian_distortion(probability=0.5, corner='bell', method='in', grid_height=4, grid_width=4, magnitude=4)
    p.skew_top_bottom(probability=0.5, magnitude=0.5)
    p.skew_left_right(probability=0.5, magnitude=0.5)
    p.skew_corner(probability=0.5, magnitude=0.5)
    p.resize(probability=1.0, width=img_size[0], height=img_size[1])  # Resize final

    print(f"[INFO] Lancement de la génération de {nb_samples} échantillons augmentés...")
    p.sample(nb_samples)

    print(f"[✅] Augmentation terminée. Fichiers disponibles dans :\n  - Images : {output_images}\n  - Masks  : {output_masks}")

    # 🎯 Tracking MLflow
    if mlflow_tracking:
        with mlflow.start_run(run_name="augmentation_pipeline"):
            mlflow.log_param("augmentation_samples", nb_samples)
            mlflow.log_param("resize_to", img_size)
            mlflow.log_param("image_input_dir", str(image_dir))
            mlflow.log_param("mask_input_dir", str(mask_dir))
            mlflow.log_param("output_dir", str(output_dir))

            # Log uniquement les chemins principaux pour audit (pas les images)
            mlflow.log_artifact(str(output_images), artifact_path="augmented_images")
            mlflow.log_artifact(str(output_masks), artifact_path="augmented_masks")
