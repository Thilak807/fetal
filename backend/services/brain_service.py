import os
import sys
import uuid
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.config import Config
    from backend.preprocessing.brain_preprocessing import BrainPreprocessor
    from backend.atlas.deformable_registration import ElasticDeformableRegistrar, get_default_atlas_template
    from backend.models.cnn_model import FetalBrainCNN
    from backend.database.db import db
    from backend.database.models import BrainImage
except (ImportError, ValueError):
    from ..config import Config
    from ..preprocessing.brain_preprocessing import BrainPreprocessor
    from ..atlas.deformable_registration import ElasticDeformableRegistrar, get_default_atlas_template
    from ..models.cnn_model import FetalBrainCNN
    from ..database.db import db
    from ..database.models import BrainImage

class BrainService:
    def __init__(self):
        self.preprocessor = BrainPreprocessor(target_size=Config.IMAGE_TARGET_SIZE)
        self.registrar = ElasticDeformableRegistrar()
        self.atlas_template = get_default_atlas_template(
            Config.ATLAS_TEMPLATES_DIR, size=Config.IMAGE_TARGET_SIZE
        )
        # Shared CNN instance for feature extraction
        self.cnn_model = FetalBrainCNN(in_channels=1, feature_dim=Config.BRAIN_FEATURE_DIM)
        self.cnn_model.eval()

    def process_brain_scan(self, file_path, patient_id=None):
        """
        Executes complete brain branch:
        Upload -> Preprocess -> Deformable Atlas Mapping -> CNN Feature Vector -> DB Record
        """
        scan_uid = f"BRAIN_{uuid.uuid4().hex[:8].upper()}"

        # 1. Load raw image
        raw_img = self.preprocessor.load_image(file_path)

        # 2. Preprocess (skull/brain ROI, intensity normalization, standard grid resize)
        preprocessed_img, prep_metrics = self.preprocessor.preprocess(raw_img)

        # 3. Save preprocessed image
        prep_filename = f"{scan_uid}_preprocessed.png"
        prep_path = os.path.join(Config.BRAIN_UPLOAD_FOLDER, prep_filename)
        Image.fromarray((preprocessed_img * 255.0).astype(np.uint8)).save(prep_path)

        # 4. Deformable Atlas Mapping (elastic registration against standard reference)
        deform_field_filename = f"{scan_uid}_deformation_field.png"
        deform_field_path = os.path.join(Config.BRAIN_UPLOAD_FOLDER, deform_field_filename)
        
        mapped_img, disp_fields, atlas_metrics = self.registrar.register(
            preprocessed_img,
            self.atlas_template,
            save_deformation_path=deform_field_path
        )

        # 5. Save atlas-mapped image
        mapped_filename = f"{scan_uid}_atlas_mapped.png"
        mapped_path = os.path.join(Config.BRAIN_UPLOAD_FOLDER, mapped_filename)
        Image.fromarray((mapped_img * 255.0).astype(np.uint8)).save(mapped_path)

        # 6. Extract spatial features using CNN
        tensor_in = torch.from_numpy(mapped_img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            f_brain = self.cnn_model.extract_features(tensor_in).squeeze(0).numpy()

        # 7. Generate Explainable AI Grad-CAM Saliency Map
        gradcam_filename = f"{scan_uid}_gradcam.png"
        gradcam_path = os.path.join(Config.BRAIN_UPLOAD_FOLDER, gradcam_filename)
        try:
            cam_map = self.cnn_model.generate_gradcam(tensor_in)
            import matplotlib.cm as cm
            heatmap_rgb = cm.jet(cam_map)[:, :, :3]
            gray_3ch = np.stack([mapped_img]*3, axis=-1)
            blended = 0.55 * gray_3ch + 0.45 * heatmap_rgb
            blended = np.clip(blended, 0.0, 1.0)
            Image.fromarray((blended * 255.0).astype(np.uint8)).save(gradcam_path)
        except Exception as e:
            print(f"[WARN] Grad-CAM generation fallback: {e}")
            Image.fromarray((mapped_img * 255.0).astype(np.uint8)).save(gradcam_path)

        feature_summary = {
            "dimension": len(f_brain),
            "mean": float(np.mean(f_brain)),
            "std": float(np.std(f_brain)),
            "l2_norm": float(np.linalg.norm(f_brain)),
            "sample_vector": [round(float(v), 4) for v in f_brain[:16]],  # Preview first 16 dims
        }

        # 8. Create or update database record if patient_id is provided
        db_record = None
        if patient_id is not None:
            db_record = BrainImage(
                image_id=scan_uid,
                patient_id=patient_id,
                raw_path=str(file_path),
                preprocessed_path=str(prep_path),
                atlas_mapped_path=str(mapped_path),
                deformation_field_path=str(deform_field_path),
                atlas_mapping=json.dumps({
                    "preprocessing": prep_metrics,
                    "atlas_registration": atlas_metrics,
                    "features": feature_summary,
                })
            )
            db.session.add(db_record)
            db.session.commit()

        return {
            "image_id": scan_uid,
            "db_id": db_record.id if db_record else None,
            "patient_id": patient_id,
            "raw_filename": os.path.basename(file_path),
            "preprocessed_filename": prep_filename,
            "atlas_mapped_filename": mapped_filename,
            "deformation_field_filename": deform_field_filename,
            "gradcam_filename": gradcam_filename,
            "preprocessed_url": f"/api/brain/image/{prep_filename}",
            "atlas_mapped_url": f"/api/brain/image/{mapped_filename}",
            "deformation_field_url": f"/api/brain/image/{deform_field_filename}",
            "gradcam_url": f"/api/brain/image/{gradcam_filename}",
            "preprocessing_metrics": prep_metrics,
            "atlas_metrics": atlas_metrics,
            "feature_summary": feature_summary,
            "status": "Atlas-Mapped & Spatial Features Extracted",
        }
