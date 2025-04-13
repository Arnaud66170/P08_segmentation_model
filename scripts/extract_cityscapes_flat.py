# scripts/extract_cityscapes_flat.py

# - Ce script te permet :
#   - d’avoir un dataset plat et propre dans data/raw/
#   - de relancer immédiatement prepare_dataset() derrière
#   - d’assurer que chaque image a son mask (base commune)

import os
import shutil
from glob import glob
from tqdm import tqdm
import sys

# ✅ Exécution du script avec arguments depuis la ligne de commande
if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise ValueError("Usage: python extract_cityscapes_flat.py <img_src> <mask_src> <output_img_dir> <output_mask_dir>")

    def extract_cityscapes_to_raw(root_img_dir, root_mask_dir, output_image_dir, output_mask_dir):
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)

        # 1. Récupération des chemins d'images et masks
        image_paths = glob(os.path.join(root_img_dir, "**", "*_leftImg8bit.png"), recursive=True)
        mask_paths = glob(os.path.join(root_mask_dir, "**", "*_gtFine_labelIds.png"), recursive=True)

        print(f"[INFO] {len(image_paths)} images trouvées")
        print(f"[INFO] {len(mask_paths)} masks trouvés")

        # 2. Dictionnaire pour faire correspondre images ↔ masks
        mask_dict = {os.path.basename(f).replace("_gtFine_labelIds.png", ""): f for f in mask_paths}

        nb_copies = 0
        for image_path in tqdm(image_paths, desc="Copie des images + masks"):
            base_name = os.path.basename(image_path).replace("_leftImg8bit.png", "")
            if base_name in mask_dict:
                new_img_name = f"{base_name}.png"
                new_mask_name = f"{base_name}.png"

                shutil.copy(image_path, os.path.join(output_image_dir, new_img_name))
                shutil.copy(mask_dict[base_name], os.path.join(output_mask_dir, new_mask_name))
                nb_copies += 1
            else:
                print(f"[WARNING] Aucun mask correspondant pour : {image_path}")

        print(f"[INFO] Total de paires copiées : {nb_copies}")

    extract_cityscapes_to_raw(
        root_img_dir=sys.argv[1],
        root_mask_dir=sys.argv[2],
        output_image_dir=sys.argv[3],
        output_mask_dir=sys.argv[4]
    )
