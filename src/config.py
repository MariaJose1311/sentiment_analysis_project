import torch
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = str(ROOT / "data" / "ResenasSiguenzaNuevo.xlsx")
MODEL_DIR = str(ROOT / "models")
OUTPUT_DIR = str(ROOT / "outputs")

# Configuración general
SEED = 42
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
