import os
import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import NUM_CLASSES

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def mean_iou(y_true, y_pred):
    # Version simplifiée pour le chargement du modèle
    return tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)(y_true, y_pred)

def load_model(model_path):
    try:
        # Charger le modèle avec les métriques personnalisées
        model = tf.keras.models.load_model(model_path, custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'mean_iou': mean_iou
        })
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        raise e