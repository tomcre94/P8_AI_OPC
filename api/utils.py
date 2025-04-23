import numpy as np
import cv2

def map_masks(mask):
    # Mapping des classes conforme à votre projet
    CLASS_MAPPING = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 9: 0, 10: 0,
        12: 0, 13: 0, 15: 0, 16: 0, 19: 0, 20: 0, 22: 0,
        25: 0, 27: 0, 32: 0,
        7: 1,   # Road
        11: 2,  # Building
        21: 3,  # Vegetation
        26: 4,  # Car
        8: 5,   # Sidewalk
        23: 6,  # Sky
        24: 7,  # Person
    }
    
    out_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    for k in CLASS_MAPPING:
        out_mask[mask == k] = CLASS_MAPPING[k]
    
    return out_mask

def preprocess_image(image, target_size):
    # Convertir BGR en RGB si nécessaire
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionner
    image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normaliser
    image = image / 255.0
    
    return image

def create_colored_mask(mask):
    # Palette de couleurs pour la visualisation
    colors = [
        [0, 0, 0],        # Background
        [128, 64, 128],   # Road
        [244, 35, 232],   # Building
        [70, 70, 70],     # Vegetation
        [107, 142, 35],   # Car
        [153, 153, 153],  # Sidewalk
        [0, 191, 255],    # Sky
        [220, 20, 60]     # Person
    ]
    
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for cls, color in enumerate(colors):
        if cls < len(colors):  # Vérifier que la classe est dans notre palette
            colored_mask[mask == cls] = color
    
    return colored_mask