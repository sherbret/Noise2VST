import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from noise2vst.models import noise2vst
    print("Import réussi !")
except Exception as e:
    print(f"Erreur lors de l'import : {e}")
