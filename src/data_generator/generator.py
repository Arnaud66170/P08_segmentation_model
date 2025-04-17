# src/data_generator/generator.py

import os
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.utils import Sequence
import albumentations as A
from albumentations.core.composition import OneOf

class AlbumentationDataGenerator(Sequence):
    """
    DataGenerator compatible Keras utilisant Albumentations pour la segmentation sémantique.
    Utilisé pour charger images + masks à la volée, avec synchronisation des augmentations.
    """

    def __init__(self, image_dir, mask_dir, batch_size=4, img_size=(256, 256),
                 augment=False, shuffle=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle

        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.mask_filenames = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        assert len(self.image_filenames) == len(self.mask_filenames), "Images et masques désalignés"

        self.on_epoch_end()

        # Définition du pipeline d’augmentation (synchronisé image/mask)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.5)
            ], p=0.5)
        ]) if self.augment else None

    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__load_batch(batch_indices)
        return np.array(X), np.array(y)

    def __load_batch(self, batch_indices):
        X_batch, y_batch = [], []

        for idx in batch_indices:
            image = cv2.imread(str(self.image_dir / self.image_filenames[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(self.mask_dir / self.mask_filenames[idx]), cv2.IMREAD_GRAYSCALE)

            # Resize
            image = cv2.resize(image, self.img_size)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            if self.augment:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            image = image.astype('float32') / 255.0
            mask = mask.astype('int32')  # Assure un encodage correct

            X_batch.append(image)
            y_batch.append(mask)

        return X_batch, y_batch