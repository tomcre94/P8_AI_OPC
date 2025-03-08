import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Définition des classes principales et de leur encodage
CLASSES = {
    "flat": 0, "human": 1, "vehicle": 2, "construction": 3,
    "object": 4, "nature": 5, "sky": 6, "void": 7
}

# Correspondance entre les classes secondaires et les classes principales
CLASS_MAPPING = {
    "road": "flat", "sidewalk": "flat", "parking": "flat", "rail track": "flat",
    "person": "human", "rider": "human",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "on rails": "vehicle", 
    "motorcycle": "vehicle", "bicycle": "vehicle", "caravan": "vehicle", "trailer": "vehicle",
    "building": "construction", "wall": "construction", "fence": "construction", 
    "guard rail": "construction", "bridge": "construction", "tunnel": "construction",
    "pole": "object", "pole group": "object", "traffic sign": "object", "traffic light": "object",
    "vegetation": "nature", "terrain": "nature",
    "sky": "sky",
    "ground": "void", "dynamic": "void", "static": "void"
}

def create_segmentation_mask(json_path, img_width, img_height, debug=False):
    """ Génère un masque de segmentation à partir d'un fichier JSON """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = np.full((img_height, img_width), 255, dtype=np.uint8)  # 255 = background/ignore
    
    print(f"Fichier JSON : {json_path}")
    print("Labels trouvés :", [obj['label'] for obj in data['objects']])

    for obj in data['objects']:
        label = obj['label']
        primary_label = CLASS_MAPPING.get(label, None)
        if primary_label and primary_label in CLASSES:
            polygon = np.array(obj['polygon'], np.int32)
            cv2.fillPoly(mask, [polygon], CLASSES[primary_label])
        elif debug:
            print(f"Label '{label}' not in CLASS_MAPPING, skipping.")
    
    return mask

def process_dataset(gtfine_dir, output_dir, debug=False):
    """ Parcourt le dossier GTFine et génère des masques """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(gtfine_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        for city in tqdm(os.listdir(split_dir), desc=f'Processing {split}'):
            city_dir = os.path.join(split_dir, city)
            output_city_dir = os.path.join(output_split_dir, city)
            os.makedirs(output_city_dir, exist_ok=True)
            
            for file in os.listdir(city_dir):
                if file.endswith('_gtFine_polygons.json'):
                    json_path = os.path.join(city_dir, file)
                    image_name = file.replace('_gtFine_polygons.json', '_gtFine_color.png')
                    
                    # Charger une image pour récupérer ses dimensions
                    img_path = os.path.join(city_dir, image_name)
                    if not os.path.exists(img_path):
                        print(f"Image not found: {img_path}")
                        continue
                    
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    
                    mask = create_segmentation_mask(json_path, img.shape[1], img.shape[0], debug)
                    
                    # Sauvegarde du masque
                    mask_path = os.path.join(output_city_dir, file.replace('_gtFine_polygons.json', '_mask.png'))
                    cv2.imwrite(mask_path, mask)
                    if debug:
                        print(f"Mask saved: {mask_path}")

# Exécution du script
gtfine_path = "./gtfine"
output_path = "./output_masks"
process_dataset(gtfine_path, output_path, debug=True)