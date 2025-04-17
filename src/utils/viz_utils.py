# src/utils/viz_utils.py

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.class_mapping import CLASS_NAME_MAPPING

def show_random_image_and_mask(img_dir, mask_dir, split="train", city="hamburg"):
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
    plt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=max(CLASS_NAME_MAPPING.keys()))
    plt.axis('off')
    plt.show()


def show_image_mask_grid_overlay(images, masks, n=5, alpha=0.5, save_path=None):
    """
    Affiche une grille de n images avec superposition de leur mask.
    Affiche aussi les noms des classes présentes dans le masque.
    """
    plt.figure(figsize=(4 * n, 6))

    for i in range(n):
        idx = random.randint(0, len(images) - 1)
        img = images[idx]
        mask = masks[idx]

        mask_display = np.ma.masked_where(mask == 255, mask)

        # Affichage de l’image + overlay
        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.imshow(mask_display, cmap="nipy_spectral", alpha=alpha)
        plt.axis("off")
        plt.title(f"Image {idx} (Overlay)")

        # Affichage du masque brut avec noms de classes
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(mask, cmap="nipy_spectral", vmin=0, vmax=max(CLASS_NAME_MAPPING.keys()))
        plt.axis("off")

        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 255]
        class_names = [CLASS_NAME_MAPPING[c] for c in unique_labels if c in CLASS_NAME_MAPPING]
        plt.title(f"Mask {idx} → {' | '.join(class_names)}")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[FIG] Figure sauvegardée dans : {save_path}")

    plt.show()

# Palette personnalisée de couleurs (8 classes)
CLASS_COLORS = {
    0: "#808080",  # void
    1: "#FFA500",  # flat
    2: "#8B4513",  # construction
    3: "#FFD700",  # object
    4: "#228B22",  # nature
    5: "#1E90FF",  # sky
    6: "#FF1493",  # human
    7: "#DC143C",  # vehicle
}




def plot_class_legend(class_mapping=CLASS_NAME_MAPPING, class_colors=CLASS_COLORS):
    """
    Affiche une légende graphique des classes avec leur couleur associée.
    """
    import matplotlib.patches as mpatches

    patches = []
    for class_id, class_name in class_mapping.items():
        color = class_colors.get(class_id, "#000000")  # fallback: noir
        patches.append(mpatches.Patch(color=color, label=f"{class_id} - {class_name}"))

    plt.figure(figsize=(8, 1.5))
    plt.legend(handles=patches, loc='center', ncol=4, frameon=False)
    plt.axis('off')
    plt.title("🎨 Légende des classes")
    plt.tight_layout()
    plt.show()


def get_custom_colormap():
    """
    Crée un ListedColormap basé sur CLASS_COLORS avec couleurs dans l’ordre d’ID.
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

    # 🔥 Correction ici : tri explicite par ID croissant
    sorted_ids = sorted(CLASS_COLORS.keys())
    palette = [hex_to_rgb(CLASS_COLORS[i]) for i in sorted_ids]
    return ListedColormap(palette)


def plot_filtered_image_mask_grid(images, masks, class_mapping, save_path=None, n=5, alpha=0.6):
    """
    Affiche une grille de n images sans masque (ligne 1)
    et leurs masques colorés uniquement (ligne 2), filtrés.
    """
    cmap = get_custom_colormap()

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))
    count = 0
    i = 0

    while count < n and i < len(images):
        img = images[i]
        mask = masks[i]
        i += 1

        # 🔧 Skip masques vides ou ignorés
        unique_vals = np.unique(mask)
        if (unique_vals == [255]).all() or (unique_vals == [0]).all():
            continue

        # 🔧 Correction d’échelle image
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0

        # 🔧 Correction type masque
        mask = mask.astype(np.uint8)

        # Affichage image brute
        axes[0, count].imshow(img)
        axes[0, count].set_title(f"Image {i}")
        axes[0, count].axis("off")

        # Affichage masque (sans 255)
        mask_display = np.ma.masked_where(mask == 255, mask)
        axes[1, count].imshow(mask_display, cmap=cmap, vmin=0, vmax=7)

        # Noms des classes présentes
        label_ids = sorted(set(np.unique(mask)) - {255})
        label_names = [class_mapping.get(l, f"unk{l}") for l in label_ids]
        axes[1, count].set_title(f"Mask {i} → {' | '.join(label_names)}")
        axes[1, count].axis("off")

        count += 1

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[FIG] Figure filtrée sauvegardée dans : {save_path}")
    plt.show()


def plot_raw_images_grid(images, save_path=None, n=5):
    """
    Affiche une grille simple d’images brutes (sans masque).
    """
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for i in range(n):
        img = images[i]

        # Normalisation si nécessaire
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")

    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[IMG] Images brutes sauvegardées dans : {save_path}")

    plt.show()