# scripts/generate_albu_dataset.py

import os
import numpy as np
from pathlib import Path
import imageio
from tqdm import tqdm

# Dossiers source et destination
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data"
processed_dir = data_dir / "processed"

# Structure cible
output_structure = {
    "train": processed_dir / "train",
    "val": processed_dir / "val",
    "test": processed_dir / "test",
}

# Créer les dossiers s'ils n'existent pas
def create_dirs(base_dir):
    for sub in ["images", "masks"]:
        path = base_dir / sub
        path.mkdir(parents=True, exist_ok=True)

# Sauvegarder chaque image et masque en .png
def save_npz_to_png(npz_path, split_name):
    dest_dir = output_structure[split_name]
    create_dirs(dest_dir)

    data = np.load(npz_path)
    images, masks = data["X"], data["Y"]

    for i in tqdm(range(len(images)), desc=f"{split_name} → PNG"):
        img_path = dest_dir / "images" / f"{i:05d}.png"
        mask_path = dest_dir / "masks" / f"{i:05d}.png"

        imageio.imwrite(img_path, (images[i] * 255).astype(np.uint8))
        imageio.imwrite(mask_path, masks[i].astype(np.uint8))

# Exécution
if __name__ == "__main__":
    npz_files = {
        "train": processed_dir / "train.npz",
        "val": processed_dir / "val.npz",
        "test": processed_dir / "test.npz",
    }

    for split, path in npz_files.items():
        if path.exists():
            save_npz_to_png(path, split)
        else:
            print(f"❌ Fichier {path.name} introuvable")
