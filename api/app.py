from flask import Flask, request, jsonify, Response
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base64
import io
import numpy as np
from flask_cors import CORS
from PIL import Image
import traceback
from werkzeug.utils import secure_filename


# Configuration des classes Cityscapes pour le mapping
CLASS_MAPPING = {
    # Classes de fond
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 9: 0, 10: 0,
    12: 0, 13: 0, 15: 0, 16: 0, 19: 0, 20: 0, 22: 0,
    25: 0, 27: 0, 32: 0,
    
    # Classes principales (par ordre d'importance/fréquence)
    7: 1,   # Road (~33%)
    11: 2,  # Building (~20%)
    21: 3,  # Vegetation (~15%)
    26: 4,  # Car (~6%)
    8: 5,   # Sidewalk (~5%)
    23: 6,  # Sky (~3.5%)
    24: 7,  # Person (~1.2%)
    
    # Classes à considérer pour un mapping futur
    28: 0,  # Bus (~0.2%)
    31: 0,  # Train (~0.1%)
    33: 0,  # Bicycle (~0.4%)
    17: 0   # Pole (~1.1%)
}

app = Flask(__name__)
CORS(app)

# Variables globales pour charger cv2 et tensorflow à la demande
cv2 = None
tf = None
model = None
_model_extracted = False

## Configuration de l'application
def extract_model_only():
    """Extrait uniquement le modèle et les fichiers Python essentiels de l'archive"""
    global _model_extracted
    if _model_extracted:
        return
        
    _model_extracted = True
    import os
    import tarfile
    
    wwwroot = "/home/site/wwwroot"
    tar_path = os.path.join(wwwroot, "output.tar.gz")
    
    # Vérifier si le modèle existe déjà
    model_dir = os.path.join(wwwroot, "models")
    model_path = os.path.join(model_dir, "unet_mini_aug_best.h5")
    
    if os.path.exists(model_path):
        print("Le modèle existe déjà, pas besoin d'extraction")
        return
        
    # Créer le dossier models s'il n'existe pas
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        print(f"Ouverture de l'archive {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            # Liste tous les fichiers dans l'archive
            all_files = tar.getnames()
            
            # Trouver le chemin du modèle dans l'archive
            model_files = [f for f in all_files if f.endswith('unet_mini_aug_best.h5')]
            if not model_files:
                print("Modèle non trouvé dans l'archive!")
                return
                
            model_file = model_files[0]
            print(f"Extraction du modèle: {model_file}")
            
            # Extraire uniquement le fichier du modèle
            model_member = tar.getmember(model_file)
            tar.extract(model_member, wwwroot)
            
            # Extraire aussi les fichiers Python essentiels
            python_files = [f for f in all_files if f.endswith('.py')]
            for py_file in python_files[:10]:  # Limiter à 10 fichiers Python pour économiser l'espace
                try:
                    print(f"Extraction de: {py_file}")
                    py_member = tar.getmember(py_file)
                    tar.extract(py_member, wwwroot)
                except Exception as e:
                    print(f"Erreur lors de l'extraction de {py_file}: {str(e)}")
            
        print("Extraction terminée avec succès")
    except Exception as e:
        print(f"Erreur lors de l'extraction: {str(e)}")
        print(traceback.format_exc())

# Exécuter l'extraction au démarrage de l'application
with app.app_context():
    try:
        extract_model_only()
    except Exception as e:
        print(f"Erreur lors de l'extraction au démarrage: {str(e)}")
        traceback.print_exc()

# S'assurer que l'extraction est tentée avant toute requête
@app.before_request
def ensure_model_extracted():
    try:
        extract_model_only()
    except Exception as e:
        print(f"Erreur lors de l'extraction avant requête: {str(e)}")
        pass  # Ne pas bloquer les requêtes en cas d'erreur

def load_dependencies():
    """Charge les dépendances lourdes uniquement à la demande"""
    global cv2, tf, model
    
    # Importer numpy explicitement
    import numpy
    import builtins
    np = numpy
    builtins.np = numpy
    
    if cv2 is None:
        print("Chargement de cv2...")
        import cv2
    
    if tf is None:
        print("Chargement de tensorflow...")
        import tensorflow as tf
        from shared.config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH
        
        # Chargement du modèle
        if model is None and os.path.exists(MODEL_PATH):
            print(f"Chargement du modèle depuis {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("Modèle chargé avec succès")
    
    return cv2, tf, model

def ensure_numpy():
    """S'assure que numpy est disponible sous le nom 'np'"""
    try:
        import sys
        if 'np' not in sys.modules:
            import numpy as np
            sys.modules['np'] = np
    except Exception as e:
        print(f"Erreur lors de la vérification de NumPy: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé qui vérifie si l'API est en ligne"""
    try:
        from shared.config import MODEL_PATH
        model_exists = os.path.exists(MODEL_PATH)
        
        # Liste des répertoires à la racine
        root_dirs = []
        try:
            root_dirs = os.listdir("/home/site/wwwroot")
        except:
            root_dirs = ["Impossible de lister le répertoire"]
        
        # Tester l'import de dépendances critiques
        dependency_status = {}
        for module in ["numpy", "cv2", "tensorflow", "PIL"]:
            try:
                __import__(module)
                dependency_status[module] = "OK"
            except Exception as e:
                dependency_status[module] = f"Erreur: {str(e)}"
        
        return jsonify({
            "status": "healthy", 
            "model": "unet_mini_lightweight", 
            "model_exists": model_exists,
            "model_path": MODEL_PATH,
            "root_dirs": root_dirs,
            "dependencies": dependency_status,
            "python_version": sys.version
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint qui prédit la segmentation d'une image"""
    try:
        ensure_numpy()
        # Charger les dépendances seulement lorsque nécessaire
        cv2, tf, loaded_model = load_dependencies()
        
        # Vérifier si l'image est présente
        if 'image' not in request.files and (not request.json or 'image' not in request.json):
            return jsonify({"success": False, "error": "Aucune image fournie"}), 400
        
        # Récupérer l'image depuis la requête
        if 'image' in request.files:
            # Depuis FormData
            file = request.files['image']
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            # Depuis JSON avec base64
            image_data = base64.b64decode(request.json['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Vérifier si l'image a été correctement chargée
        if image is None:
            return jsonify({"success": False, "error": "Impossible de décoder l'image"}), 400
        
        # Convertir BGR en RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner l'image
        from shared.config import IMG_HEIGHT, IMG_WIDTH
        resized_img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normaliser l'image
        processed_img = resized_img / 255.0
        
        # Prédiction
        prediction = loaded_model.predict(np.expand_dims(processed_img, axis=0))[0]
        pred_mask = np.argmax(prediction, axis=-1)
        
        # Créer un masque coloré
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
        
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for cls, color in enumerate(colors):
            colored_mask[pred_mask == cls] = color
            
        # Convertir en image PNG
        img_pil = Image.fromarray(colored_mask)
        img_io = io.BytesIO()
        img_pil.save(img_io, 'PNG')
        img_io.seek(0)
        
        # Encoder en base64 pour la réponse JSON
        mask_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "mask_base64": mask_base64,
            "classes_found": np.unique(pred_mask).tolist()
        })
        
    except Exception as e:
        # Log l'erreur complète pour le débogage
        print(f"Erreur lors de la prédiction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Erreur lors du traitement: {str(e)}"
        }), 500

@app.route('/predict_with_mask', methods=['POST'])
def predict_with_mask():
    """Endpoint qui prédit la segmentation et compare avec un masque réel"""
    try:
        # Vérifier d'abord que les fichiers ont été fournis
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({
                "success": False, 
                "error": "L'image et le masque sont requis"
            }), 400
        
        # Récupérer les fichiers
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        print(f"Fichier image reçu: {image_file.filename}, type: {image_file.content_type}")
        print(f"Fichier masque reçu: {mask_file.filename}, type: {mask_file.content_type}")
        
        # Charger les dépendances seulement lorsque nécessaire
        cv2, tf, loaded_model = load_dependencies()
                
        # Traiter l'image
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"success": False, "error": "Impossible de décoder l'image"}), 400
            
        # Traiter le masque
        mask_in_memory = io.BytesIO()
        mask_file.save(mask_in_memory)
        mask_in_memory.seek(0)  # Important: revenir au début du flux

        # Utiliser PIL pour charger le masque et préserver le type original
        mask_pil = Image.open(mask_in_memory)
        mask = np.array(mask_pil)

        print(f"Format du masque chargé: {mask_pil.mode}, dtype: {mask.dtype}")
        print(f"Dimensions du masque: {mask.shape}")

        # Afficher les valeurs uniques dans le masque pour le débogage
        unique_mask_values = np.unique(mask)
        print(f"Valeurs uniques dans le masque original: {unique_mask_values}")

        # Si le masque est en mode palette, convertir en raster
        if mask_pil.mode == 'P':
            print("Conversion du masque en mode palette vers raster...")
            mask_pil = mask_pil.convert('L')
            mask = np.array(mask_pil)
            print(f"Nouvelles valeurs uniques après conversion: {np.unique(mask)}")
        
        # Charger la configuration
        from shared.config import IMG_HEIGHT, IMG_WIDTH
        
        # Définir notre propre fonction de mapping ici pour éviter les problèmes d'importation
        def cityscapes_to_model_classes(cityscapes_mask):
            """
            Convertit les ID des masques Cityscapes vers les 8 classes du modèle
            en utilisant le même mapping que celui utilisé pendant l'entraînement
            """
            # Créer un masque vide avec la même forme
            model_mask = np.zeros_like(cityscapes_mask, dtype=np.uint8)
            
            # Afficher les informations de débogage
            unique_vals = np.unique(cityscapes_mask)
            print(f"Masque format: {cityscapes_mask.dtype}, plage: {cityscapes_mask.min()}-{cityscapes_mask.max()}")
            print(f"Valeurs uniques dans le masque original: {unique_vals}")
            
            # Appliquer le mapping exactement comme pendant l'entraînement
            for k in CLASS_MAPPING:
                model_mask[cityscapes_mask == k] = CLASS_MAPPING[k]
            
            # Vérifier les résultats du mapping
            mapped_vals = np.unique(model_mask)
            print(f"Valeurs uniques après mapping: {mapped_vals}")
            
            # Si aucune classe n'est mappée (masque entièrement noir), essayer une stratégie alternative
            if len(mapped_vals) <= 1 and mapped_vals[0] == 0:
                print("ATTENTION: Masque entièrement noir après mapping, utilisation du mapping de secours")
                # Essayer une approche plus générique basée sur les plages de valeurs
                model_mask[(cityscapes_mask > 0) & (cityscapes_mask < 10)] = 1  # Route/Sidewalk
                model_mask[(cityscapes_mask >= 10) & (cityscapes_mask < 20)] = 2  # Building
                model_mask[(cityscapes_mask >= 20) & (cityscapes_mask < 25)] = 3  # Vegetation
                model_mask[(cityscapes_mask >= 25) & (cityscapes_mask < 30)] = 4  # Car
                model_mask[(cityscapes_mask >= 30) & (cityscapes_mask < 40)] = 7  # Person
                
                # Réafficher les valeurs après stratégie alternative
                backup_vals = np.unique(model_mask)
                print(f"Valeurs après mapping de secours: {backup_vals}")
            
            return model_mask
        
        if mask is None:
            return jsonify({"success": False, "error": "Impossible de décoder le masque"}), 400

        # Visualiser le masque brut pour le débogage
        raw_mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        # Normaliser le masque pour la visualisation (étire les valeurs entre 0-255)
        if mask.max() > 0:  # Éviter la division par zéro
            normalized = (mask.astype(float) * 255 / mask.max()).astype(np.uint8)
            raw_mask_vis[..., 0] = normalized  # Canal rouge
            raw_mask_vis[..., 1] = normalized  # Canal vert
            raw_mask_vis[..., 2] = normalized  # Canal bleu

        # Convertir en base64 pour la visualisation
        raw_mask_pil = Image.fromarray(raw_mask_vis)
        raw_mask_io = io.BytesIO()
        raw_mask_pil.save(raw_mask_io, 'PNG')
        raw_mask_io.seek(0)
        raw_mask_base64 = base64.b64encode(raw_mask_io.getvalue()).decode('utf-8')

        # Convertir BGR en RGB pour l'image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner l'image et le masque
        resized_img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        resized_mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # Mapper les classes du masque avec notre nouvelle fonction
        mapped_mask = cityscapes_to_model_classes(resized_mask)
        
        # Vérifier les valeurs du masque mappé
        unique_mapped_values = np.unique(mapped_mask)
        print(f"Valeurs uniques dans le masque mappé: {unique_mapped_values}")
        
        # Normaliser l'image pour la prédiction
        processed_img = resized_img / 255.0
        
        # Prédiction
        prediction = loaded_model.predict(np.expand_dims(processed_img, axis=0))[0]
        pred_mask = np.argmax(prediction, axis=-1)
        
        # Couleurs pour visualisation
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
        
        # Créer des masques colorisés
        colored_pred_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        colored_real_mask = np.zeros((*mapped_mask.shape, 3), dtype=np.uint8)
        # Application des couleurs pour chaque classe
        for cls, color in enumerate(colors):
            # Vérifier si la classe existe dans le masque avant de l'appliquer
            if cls in np.unique(pred_mask):
                colored_pred_mask[pred_mask == cls] = color
            if cls in np.unique(mapped_mask):
                colored_real_mask[mapped_mask == cls] = color
        for cls, color in enumerate(colors):
            colored_pred_mask[pred_mask == cls] = color
            colored_real_mask[mapped_mask == cls] = color
                       
        # Convertir en images PNG puis en base64
        pred_mask_pil = Image.fromarray(colored_pred_mask)
        real_mask_pil = Image.fromarray(colored_real_mask)
        
        pred_img_io = io.BytesIO()
        real_img_io = io.BytesIO()
        
        pred_mask_pil.save(pred_img_io, 'PNG')
        real_mask_pil.save(real_img_io, 'PNG')
        
        pred_img_io.seek(0)
        real_img_io.seek(0)
        
        pred_mask_base64 = base64.b64encode(pred_img_io.getvalue()).decode('utf-8')
        real_mask_base64 = base64.b64encode(real_img_io.getvalue()).decode('utf-8')
        
        # Calculer des métriques de comparaison
        # Note: On utilise la classe comme index pour la comparaison
        accuracy = np.mean(pred_mask == mapped_mask)
        
        # Calcul de l'IoU pour chaque classe et moyenne
        iou_per_class = {}
        mean_iou = 0
        num_classes = 0
        
        for cls in range(8):  # Pour les 8 classes
            true_class = (mapped_mask == cls)
            pred_class = (pred_mask == cls)
            
            intersection = np.logical_and(true_class, pred_class).sum()
            union = np.logical_or(true_class, pred_class).sum()
            
            if union > 0:
                iou = intersection / union
                iou_per_class[int(cls)] = float(iou)
                mean_iou += iou
                num_classes += 1
        
        if num_classes > 0:
            mean_iou = mean_iou / num_classes
        
        return jsonify({
            "success": True,
            "pred_mask_base64": pred_mask_base64,
            "real_mask_base64": real_mask_base64,
            "raw_mask_base64": raw_mask_base64,  # Ajouté pour le débogage
            "unique_mask_values": unique_mask_values.tolist(),  # Valeurs uniques pour débogage
            "metrics": {
                "iou": float(mean_iou),
                "iou_per_class": iou_per_class,
                "accuracy": float(accuracy)
            }
        })
        
    except Exception as e:
        print(f"Erreur dans predict_with_mask: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    """Vérifier l'état du modèle"""
    try:
        from shared.config import MODEL_PATH
        model_exists = os.path.exists(MODEL_PATH)
        model_path = MODEL_PATH
        
        # Vérifier si le dossier parent existe
        parent_dir = os.path.dirname(MODEL_PATH)
        parent_exists = os.path.exists(parent_dir)
        
        # Lister les fichiers dans le dossier parent s'il existe
        parent_contents = os.listdir(parent_dir) if parent_exists else []
        
        # Vérifier si TensorFlow peut être importé
        tf_imported = False
        tf_version = None
        try:
            import tensorflow as tf
            tf_imported = True
            tf_version = tf.__version__
        except Exception as tf_err:
            tf_version = f"Error: {str(tf_err)}"
        
        return jsonify({
            "success": True,
            "model_exists": model_exists,
            "model_path": model_path,
            "parent_dir_exists": parent_exists,
            "parent_dir_contents": parent_contents,
            "tensorflow_imported": tf_imported,
            "tensorflow_version": tf_version,
            "python_version": sys.version
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        })

@app.errorhandler(Exception)
def handle_exception(e):
    """Gère toutes les exceptions non capturées et renvoie un JSON"""
    print(f"Erreur non gérée: {str(e)}")
    print(traceback.format_exc())
    return jsonify({
        "success": False,
        "error": str(e),
        "trace": traceback.format_exc()
    }), 500

@app.route('/', methods=['GET'])
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>U-Net Segmentation Demo</title>
        <style>
            body { font-family: Arial; max-width: 1000px; margin: 0 auto; padding: 20px; }
            .container { display: flex; flex-wrap: wrap; }
            .dropzone { border: 2px dashed #0087F7; border-radius: 5px; padding: 20px; text-align: center; margin: 10px 0; }
            .preview-container { display: flex; justify-content: space-between; margin-top: 20px; }
            .preview-box { flex: 1; margin: 0 10px; text-align: center; border: 1px dashed #ccc; padding: 10px; }
            #results { display: flex; justify-content: space-between; margin-top: 20px; }
            .result-image { flex: 1; margin: 0 5px; text-align: center; }
            button { background: #0087F7; color: white; border: none; padding: 10px 20px; margin: 10px 0; cursor: pointer; }
            button:hover { background: #0077d8; }
            .loader { border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 2s linear infinite; margin: 20px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .dropzone-section { flex: 1; margin: 0 10px; }
            .upload-section { display: flex; flex-wrap: wrap; }
            .instructions { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .instructions code { background-color: #e9ecef; padding: 2px 5px; border-radius: 3px; }
            .results-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 30px; }
            .result-card { border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; }
            .result-card h3 { margin-top: 0; text-align: center; }
            .color-legend { display: flex; flex-wrap: wrap; margin-top: 20px; border-top: 1px solid #dee2e6; padding-top: 15px; }
            .color-item { display: flex; align-items: center; margin: 5px 15px; }
            .color-box { width: 20px; height: 20px; margin-right: 10px; }
        </style>
    </head>
    <body>
        <h1>U-Net Segmentation Demo</h1>
        
        <div class="instructions">
            <h2>Instructions</h2>
            <p>Cette application permet de comparer la prédiction du modèle U-Net avec un masque de vérité terrain.</p>
            <ol>
                <li>Téléchargez une image urbaine (format PNG ou JPG)</li>
                <li>Téléchargez le masque de vérité terrain correspondant (format <code>*_gtFine_labelIds.png</code>)</li>
                <li>Cliquez sur "Lancer la prédiction" pour voir la segmentation générée par le modèle</li>
            </ol>
        </div>

        <div class="upload-section">
            <div class="dropzone-section">
                <h3>Image</h3>
                <div class="dropzone" id="image-dropzone">
                    <p>Glissez-déposez une image ici ou cliquez pour sélectionner</p>
                    <input type="file" id="imageInput" style="display: none;" accept="image/*">
                </div>
                <div class="preview-box">
                    <img id="originalImage" style="max-width: 100%; max-height: 200px; display: none;">
                </div>
            </div>
            
            <div class="dropzone-section">
                <h3>Masque de vérité terrain</h3>
                <div class="dropzone" id="mask-dropzone">
                    <p>Glissez-déposez un masque ici ou cliquez pour sélectionner</p>
                    <input type="file" id="maskInput" style="display: none;" accept="image/png">
                </div>
                <div class="preview-box">
                    <img id="originalMask" style="max-width: 100%; max-height: 200px; display: none;">
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <button id="predictBtn" disabled>Lancer la prédiction</button>
            <div class="loader" id="loader"></div>
        </div>
        
        <div class="results-grid">
            <div class="result-card">
                <img id="resultOriginal" style="max-width: 100%; max-height: 300px; display: none;">
                <h3>Image originale</h3>
            </div>
            <div class="result-card">
                <img id="resultRealMask" style="max-width: 100%; max-height: 300px; display: none;">
                <h3>Masque réel</h3>
            </div>
            <div class="result-card">
                <img id="resultPredictedMask" style="max-width: 100%; max-height: 300px; display: none;">
                <h3>Masque prédit</h3>
            </div>
        </div>
        
        <div class="color-legend">
            <h3>Légende des classes</h3>
            <div style="display: flex; flex-wrap: wrap; width: 100%;">
                <div class="color-item"><div class="color-box" style="background-color: rgb(0,0,0);"></div>Fond</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(128,64,128);"></div>Route</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(244,35,232);"></div>Bâtiment</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(70,70,70);"></div>Végétation</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(107,142,35);"></div>Voiture</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(153,153,153);"></div>Trottoir</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(0,191,255);"></div>Ciel</div>
                <div class="color-item"><div class="color-box" style="background-color: rgb(220,20,60);"></div>Personne</div>
            </div>
        </div>

        <script>
            // Éléments DOM pour l'image
            const imageDropzone = document.getElementById('image-dropzone');
            const imageInput = document.getElementById('imageInput');
            const originalImage = document.getElementById('originalImage');
            
            // Éléments DOM pour le masque
            const maskDropzone = document.getElementById('mask-dropzone');
            const maskInput = document.getElementById('maskInput');
            const originalMask = document.getElementById('originalMask');
            
            // Éléments DOM communs
            const predictBtn = document.getElementById('predictBtn');
            const loader = document.getElementById('loader');
            const resultOriginal = document.getElementById('resultOriginal');
            const resultRealMask = document.getElementById('resultRealMask');
            const resultPredictedMask = document.getElementById('resultPredictedMask');
            
            // Variables pour stocker les fichiers
            let currentImageFile = null;
            let currentMaskFile = null;
            
            // Configuration du drag and drop pour l'image
            setupDropzone(imageDropzone, imageInput, handleImageFile);
            
            // Configuration du drag and drop pour le masque
            setupDropzone(maskDropzone, maskInput, handleMaskFile);
            
            // Fonction pour configurer une zone de glisser-déposer
            function setupDropzone(dropzone, input, handleFileFunc) {
                dropzone.addEventListener('click', () => input.click());
                
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    dropzone.addEventListener(eventName, preventDefaults, false);
                });
                
                ['dragenter', 'dragover'].forEach(eventName => {
                    dropzone.addEventListener(eventName, () => {
                        dropzone.style.borderColor = '#0077d8';
                        dropzone.style.backgroundColor = '#f0f8ff';
                    }, false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    dropzone.addEventListener(eventName, () => {
                        dropzone.style.borderColor = '#0087F7';
                        dropzone.style.backgroundColor = '';
                    }, false);
                });
                
                dropzone.addEventListener('drop', (e) => {
                    const dt = e.dataTransfer;
                    const files = dt.files;
                    if (files.length) {
                        handleFileFunc(files[0]);
                    }
                }, false);
                
                input.addEventListener('change', (e) => {
                    if (e.target.files.length) {
                        handleFileFunc(e.target.files[0]);
                    }
                }, false);
            }
            
            // Prévenir le comportement par défaut
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            // Gérer le fichier image
            function handleImageFile(file) {
                currentImageFile = file;
                displayPreview(file, originalImage);
                checkEnablePredictButton();
            }
            
            // Gérer le fichier de masque
            function handleMaskFile(file) {
                currentMaskFile = file;
                displayPreview(file, originalMask);
                checkEnablePredictButton();
            }
            
            // Afficher l'aperçu d'un fichier
            function displayPreview(file, imgElement) {
                if (file.type.match('image.*')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imgElement.src = e.target.result;
                        imgElement.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            }
            
            // Activer le bouton de prédiction si les deux fichiers sont disponibles
            function checkEnablePredictButton() {
                predictBtn.disabled = !(currentImageFile && currentMaskFile);
            }
            
            // Gestionnaire pour le bouton de prédiction
            predictBtn.addEventListener('click', async () => {
                if (!currentImageFile || !currentMaskFile) return;
                
                loader.style.display = 'block';
                predictBtn.disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('image', currentImageFile);
                    formData.append('mask', currentMaskFile);
                    
                    console.log("Envoi des fichiers à l'API...");
                    const response = await fetch('/predict_with_mask', {
                        method: 'POST',
                        body: formData
                    });
                    
                    console.log("Réponse reçue:", response.status);
                    const text = await response.text();
                    console.log("Contenu brut:", text);
                    
                    // Essayer de parser en JSON
                    let data;
                    try {
                        data = JSON.parse(text);
                    } catch (e) {
                        throw new Error(`Échec du parsing JSON: ${text}`);
                    }
                    
                    if (data.success) {
                        // Afficher l'image originale
                        resultOriginal.src = originalImage.src;
                        resultOriginal.style.display = 'block';
                        
                        // Afficher le masque réel colorisé
                        resultRealMask.src = `data:image/png;base64,${data.real_mask_base64}`;
                        resultRealMask.style.display = 'block';
                        
                        // Afficher le masque prédit
                        resultPredictedMask.src = `data:image/png;base64,${data.pred_mask_base64}`;
                        resultPredictedMask.style.display = 'block';
                        
                        // Afficher les statistiques de performance (facultatif)
                        if (data.metrics) {
                            console.log("Métriques:", data.metrics);
                        }
                    } else {
                        alert(`Erreur: ${data.error || 'Échec de la prédiction'}`);
                    }
                } catch (error) {
                    console.error('Erreur:', error);
                    alert(`Erreur lors de la communication avec l'API: ${error.message}`);
                } finally {
                    loader.style.display = 'none';
                    predictBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return html
@app.route('/analyze_mask', methods=['POST'])
def analyze_mask():
    """Endpoint pour analyser un masque sans faire de prédiction"""
    try:
        if 'mask' not in request.files:
            return jsonify({"success": False, "error": "Masque requis"}), 400
            
        mask_file = request.files['mask']
        
        # Tenter plusieurs méthodes de chargement
        # 1. Méthode cv2
        mask_in_memory = io.BytesIO()
        mask_file.save(mask_in_memory)
        mask_data = np.frombuffer(mask_in_memory.getvalue(), dtype=np.uint8)
        cv2_mask = cv2.imdecode(mask_data, cv2.IMREAD_UNCHANGED)
        
        # 2. Méthode PIL
        mask_file.seek(0)  # Revenir au début du fichier
        pil_mask = Image.open(mask_file)
        pil_array = np.array(pil_mask)
        
        return jsonify({
            "success": True,
            "cv2_analysis": {
                "shape": cv2_mask.shape if cv2_mask is not None else None,
                "dtype": str(cv2_mask.dtype) if cv2_mask is not None else None,
                "unique_values": np.unique(cv2_mask).tolist() if cv2_mask is not None else None
            },
            "pil_analysis": {
                "shape": pil_array.shape,
                "dtype": str(pil_array.dtype),
                "mode": pil_mask.mode,
                "unique_values": np.unique(pil_array).tolist()
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))