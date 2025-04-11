# 🔧 Outils système
import os
import sys

# 📊 Traitement de données
import numpy as np
import pandas as pd
from tqdm import tqdm

# 🖼️ Traitement d’images
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

# 📚 Machine Learning / Deep Learning
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 🌐 API & UI
import fastapi
import uvicorn
import gradio as gr

# 🔁 Suivi de modèle (tracking)
import mlflow

# ⚙️ Environnement
from dotenv import load_dotenv
load_dotenv()

# 🧪 Tests
import pytest
