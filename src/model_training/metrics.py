# src/model_training/metrics.py

import tensorflow as tf
from tensorflow.keras import backend as K

def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union pour la segmentation.
    """
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true_f = K.flatten(tf.one_hot(y_true, tf.shape(y_pred)[-1]))
    y_pred_f = K.flatten(tf.one_hot(y_pred, tf.shape(y_pred)[-1]))

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient : 2 * intersection / (sum préd + sum vrai)
    """
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)

    y_true_f = K.flatten(tf.one_hot(y_true, tf.shape(y_pred)[-1]))
    y_pred_f = K.flatten(tf.one_hot(y_pred, tf.shape(y_pred)[-1]))

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
