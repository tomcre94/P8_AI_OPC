import os

# Configuration du modèle
IMG_HEIGHT = 128
IMG_WIDTH = 256
NUM_CLASSES = 8

# Définition des classes à considérer
CLASS_MAPPING = {
    # Classes de fond
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 9: 0, 10: 0,
    12: 0, 13: 0, 15: 0, 16: 0, 19: 0, 20: 0, 22: 0,
    25: 0, 27: 0, 32: 0,
    
    # Classes principales
    7: 1,   # Road
    11: 2,  # Building
    21: 3,  # Vegetation
    26: 4,  # Car
    8: 5,   # Sidewalk
    23: 6,  # Sky
    24: 7,  # Person
}

def map_masks(mask):
    """Convertit les IDs du masque original vers les nouvelles classes"""
    import numpy as np
    out_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    # Mapper les classes connues
    for k in CLASS_MAPPING:
        out_mask[mask == k] = CLASS_MAPPING[k]
    
    return out_mask

def find_model_path():
    """Recherche le modèle dans différents emplacements possibles"""
    import os
    import glob
    
    # Chemins possibles où le modèle pourrait se trouver
    search_paths = [
        # Chemin original
        os.path.join("/home/site/wwwroot", "models", "unet_mini_aug_best.h5"),
        # Chemins après extraction ciblée
        os.path.join("/home/site/wwwroot", "unet_mini_aug_best.h5")
    ]
    
    # Chercher dans les chemins explicites
    for path in search_paths:
        if os.path.exists(path):
            print(f"Modèle trouvé à: {path}")
            return path
    
    # Recherche récursive en dernier recours
    try:
        for root, dirs, files in os.walk("/home/site/wwwroot"):
            for file in files:
                if file.endswith(".h5"):
                    path = os.path.join(root, file)
                    print(f"Modèle trouvé après recherche à: {path}")
                    return path
    except Exception as e:
        print(f"Erreur lors de la recherche récursive: {str(e)}")
    
    # Si rien n'est trouvé, retourner le chemin par défaut
    default_path = os.path.join("/home/site/wwwroot", "models", "unet_mini_aug_best.h5")
    print(f"Aucun modèle trouvé, utilisation du chemin par défaut: {default_path}")
    return default_path

# Remplacer la définition de MODEL_PATH par ceci:
if os.environ.get("WEBSITE_SITE_NAME"):
    MODEL_PATH = find_model_path()
else:
    MODEL_PATH = os.environ.get("MODEL_PATH", 
                              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "models", "unet_mini_aug_best.h5"))