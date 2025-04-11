# src/utils/viz_utils.py

import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def show_random_image_and_mask(img_dir, mask_dir, split="train", city="hamburg"):
    """
    Affiche une image et son mask correspondant, choisis aléatoirement pour une ville et un split donnés.

    :param img_dir: Path vers le dossier contenant leftImg8bit
    :param mask_dir: Path vers le dossier contenant gtFine
    :param split: "train", "val" ou "test"
    :param city: nom du sous-dossier (ville)
    """
    img_path = img_dir / split / city
    mask_path = mask_dir / split / city

    img_files = sorted(img_path.glob("*_leftImg8bit.png"))
    mask_files = sorted(mask_path.glob("*_gtFine_labelIds.png"))

    if not img_files or not mask_files:
        print("Aucune image ou mask trouvé.")
        return

    idx = random.randint(0, len(img_files) - 1)

    img = cv2.imread(str(img_files[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask (classes)")
    plt.imshow(mask, cmap='nipy_spectral')
    plt.axis('off')
    plt.show()
