import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_sample_mask(mask_path):
    """ Affiche un exemple de masque généré pour vérification """
    if not os.path.exists(mask_path):
        print(f"Mask file not found: {mask_path}")
        return
    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Failed to read mask: {mask_path}")
        return
    
    unique_values = np.unique(mask)
    print(f"Valeurs uniques dans le masque : {unique_values}")
    plt.imshow(mask, cmap="nipy_spectral")
    plt.title("Exemple de masque généré")
    plt.colorbar()
    plt.show()

# Vérifier un exemple de masque
example_mask = ".\output_masks/test/munich/munich_000041_000019_mask.png"
display_sample_mask(example_mask)