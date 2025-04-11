# src/data_preprocessing/preprocessing.py

import cv2
import numpy as np
from typing import Tuple

def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensionne une image RGB à la taille souhaitée.

    :param img: image (np.ndarray)
    :param target_size: (largeur, hauteur)
    :return: image redimensionnée
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensionne un masque (np.ndarray) avec interpolation NEAREST (important pour les classes).

    :param mask: masque (np.ndarray)
    :param target_size: (largeur, hauteur)
    :return: mask redimensionné
    """
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalise une image en pixels flottants entre 0 et 1.

    :param img: image (np.ndarray)
    :return: image normalisée
    """
    return img.astype('float32') / 255.0


def one_hot_encode_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Encode un masque en one-hot selon le nombre de classes.

    :param mask: masque 2D (H, W)
    :param num_classes: nombre de classes
    :return: masque one-hot (H, W, C)
    """
    return np.eye(num_classes)[mask].astype('float32')
