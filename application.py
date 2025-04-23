import sys
import os

# Chemin absolu vers le répertoire API (corrigé pour la nouvelle structure)
api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api')
sys.path.insert(0, api_dir)

# Importer l'application Flask depuis api/app.py
from app import app as application

# Point d'entrée pour Gunicorn
app = application

if __name__ == "__main__":
    app.run()