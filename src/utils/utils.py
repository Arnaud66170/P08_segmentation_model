# src/utils/utils.py

import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from utils.logger import log_step
import matplotlib.pyplot as plt
# os.makedirs(os.path.dirname(output_path), exist_ok=True)


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Ignore les labels 255 (classe à ignorer) dans le calcul de la loss.
    y_true doit être squeezé pour correspondre à [batch, H, W]
    """
    y_true = tf.squeeze(y_true, axis=-1)  # ⚠️ suppression de la 4e dimension inutile
    mask = tf.not_equal(y_true, 255)
    mask = tf.cast(mask, dtype=tf.float32)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    loss = loss * mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

@log_step
def plot_history(history, output_path):
    # Protection contre historique vide ou de test
    if not hasattr(history, 'history') or not history.history:
        print("[WARN] ⛔ Aucun historique d'entraînement à tracer.")
        return

    hist = history.history

    if "loss" not in hist:
        print("[WARN] ⛔ Aucune courbe 'loss' trouvée dans l'historique.")
        return

    plt.figure(figsize=(10, 4))

    # Courbe des pertes
    plt.subplot(1, 2, 1)
    plt.plot(hist['loss'], label='loss')
    plt.plot(hist.get('val_loss', []), label='val_loss')
    plt.title("Loss")
    plt.legend()

    # Courbe des accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist.get('accuracy', []), label='acc')
    plt.plot(hist.get('val_accuracy', []), label='val_acc')
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Création du dossier si nécessaire
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    # ✅ Affichage interactif uniquement si dans un notebook
    if os.environ.get('IPYTHONENABLE', '1') == '1':
        plt.show()

    plt.close()