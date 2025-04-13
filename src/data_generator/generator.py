# src/data_generator/generator.py

import numpy as np
import albumentations as A
from tensorflow.keras.utils import Sequence
import cv2

class CityscapesDataGenerator(Sequence):
    """
    Générateur de données pour segmentation sémantique.
    Il prend des arrays en entrée (X = images, Y = masks), et renvoie des batchs augmentés.
    Compatible avec les modèles Keras. Utilise Albumentations pour la data augmentation.
    """

    def __init__(self, X, Y, batch_size=16, img_size=(256, 256), augment=False, shuffle=True):
        """
        Initialise le générateur

        :param X: array-like ou liste des images (déjà chargées ou paths)
        :param Y: array-like ou liste des masks (déjà chargés ou paths)
        :param batch_size: taille des batchs
        :param img_size: taille des images en entrée du modèle (H, W)
        :param augment: booléen, si True applique des transformations Albumentations
        :param shuffle: booléen, mélange les données à chaque epoch
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(X))

        # Définition des augmentations (on peut personnaliser à volonté)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
        ]) if augment else None

        self.on_epoch_end()

    def __len__(self):
        """
        Retourne le nombre total de batchs par epoch
        """
        return int(np.ceil(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        """
        Mélange les indices à la fin de chaque epoch si shuffle est activé
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Renvoie un batch indexé
        """
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = []
        batch_Y = []

        for i in batch_indices:
            image = self.X[i]
            mask = self.Y[i]

            # Resize (sécurité)
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            # Appliquer augmentation (si activée)
            if self.augment and self.transform:
                image = image.astype(np.float32)  # ⚠️ indispensable pour Albumentations
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']


            # Ajout dans la liste du batch
            batch_X.append(image)
            batch_Y.append(mask)

        # Conversion en tableaux numpy
        batch_X = np.array(batch_X, dtype=np.float32)
        batch_Y = np.array(batch_Y, dtype=np.uint8)

        # Normalisation [0, 1]
        batch_X /= 255.0

        # Ajouter canal au mask si nécessaire (pour modèle à sortie binaire par pixel)
        if len(batch_Y.shape) == 3:
            batch_Y = np.expand_dims(batch_Y, axis=-1)

        return batch_X, batch_Y
