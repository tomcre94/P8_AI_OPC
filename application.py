import sys
import os

# Ajouter le chemin api au path Python
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))

# Importer l'app Flask depuis api/app.py
from api.app import app

# Pour que Gunicorn puisse la trouver
if __name__ == "__main__":
    app.run()