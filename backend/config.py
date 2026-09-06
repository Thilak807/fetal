import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get("SECRET_KEY", "fetal-multimodal-secret-key-2026")
    
    # Paths
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_FOLDER = BASE_DIR / "backend" / "uploads"
    BRAIN_UPLOAD_FOLDER = UPLOAD_FOLDER / "brain"
    CARDIAC_UPLOAD_FOLDER = UPLOAD_FOLDER / "cardiac"
    SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
    ATLAS_TEMPLATES_DIR = BASE_DIR / "backend" / "atlas" / "templates"
    SAVED_MODELS_DIR = BASE_DIR / "backend" / "saved_models"
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'fetal_health.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File limits
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "dcm", "nii", "gz"}
    ALLOWED_SIGNAL_EXTENSIONS = {"csv", "dat", "txt", "hea"}

    # Signal & Image Constants
    CARDIAC_SAMPLING_RATE = 4  # 4 Hz (CTU-CHB standard)
    CARDIAC_WINDOW_LEN = 120    # 120 points = 30 seconds at 4 Hz
    IMAGE_TARGET_SIZE = (128, 128)
    
    # Model dimensions
    BRAIN_FEATURE_DIM = 128
    CARDIAC_FEATURE_DIM = 128
    FUSION_HIDDEN_DIM = 128
    NUM_CLASSES = 3  # Normal, Suspect, Pathological (or 2 for Hypoxia Risk)
    
    # Clinical disclaimer
    DISCLAIMER = (
        "ACADEMIC RESEARCH PROTOTYPE: This AI-assisted system is designed for multi-modal "
        "fetal risk research and is NOT approved for independent clinical diagnosis. "
        "All predictions must be verified by a qualified medical professional."
    )

# Ensure directories exist
for p in [
    Config.UPLOAD_FOLDER,
    Config.BRAIN_UPLOAD_FOLDER,
    Config.CARDIAC_UPLOAD_FOLDER,
    Config.SAMPLE_DATA_DIR,
    Config.ATLAS_TEMPLATES_DIR,
    Config.SAVED_MODELS_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
