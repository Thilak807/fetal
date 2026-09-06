import os
import sys
import uuid
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.config import Config
    from backend.models.fusion_model import MultiModalFusionModel
    from backend.database.db import db
    from backend.database.models import Patient, BrainImage, CardiacSignal, RiskAssessment
except (ImportError, ValueError):
    from ..config import Config
    from ..models.fusion_model import MultiModalFusionModel
    from ..database.db import db
    from ..database.models import Patient, BrainImage, CardiacSignal, RiskAssessment

class PredictionService:
    def __init__(self):
        self.fusion_model = MultiModalFusionModel(
            brain_feature_dim=Config.BRAIN_FEATURE_DIM,
            cardiac_feature_dim=Config.CARDIAC_FEATURE_DIM,
            fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
            use_gated_attention=True,
        )
        # Attempt to load trained weights if available
        weights_path = os.path.join(Config.SAVED_MODELS_DIR, "multimodal_fusion.pt")
        if os.path.exists(weights_path):
            try:
                self.fusion_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            except Exception as e:
                print(f"[WARN] Could not load saved fusion model: {e}")
        self.fusion_model.eval()

    def predict_multimodal(self, patient_id, brain_image_id=None, cardiac_signal_id=None, clinician_notes=None):
        """
        Executes end-to-end multi-modal feature fusion and risk assessment:
        [Atlas-Mapped Brain Image] + [Conditioned Cardiac Sequence]
                 |                              |
                 v                              v
            Brain CNN                      Cardiac LSTM
                 |                              |
              F_brain                        F_cardiac
                 +--------------+---------------+
                                |
                                v
                       Feature-Level Fusion
                                |
                                v
                         Risk Assessment
        """
        patient = Patient.query.get(patient_id)
        if not patient:
            raise ValueError(f"Patient with ID {patient_id} not found.")

        # 1. Retrieve brain image representation
        brain_rec = None
        if brain_image_id:
            brain_rec = BrainImage.query.get(brain_image_id)
        elif patient.brain_images:
            brain_rec = patient.brain_images[-1]

        if not brain_rec or not os.path.exists(brain_rec.atlas_mapped_path):
            raise ValueError("No atlas-mapped brain image available for this assessment.")

        with Image.open(brain_rec.atlas_mapped_path) as img:
            mapped_array = np.array(img.convert('L'), dtype=np.float32) / 255.0
        brain_tensor = torch.from_numpy(mapped_array).unsqueeze(0).unsqueeze(0).float()

        # 2. Retrieve cardiac signal representation
        cardiac_rec = None
        if cardiac_signal_id:
            cardiac_rec = CardiacSignal.query.get(cardiac_signal_id)
        elif patient.cardiac_signals:
            cardiac_rec = patient.cardiac_signals[-1]

        if not cardiac_rec or not os.path.exists(cardiac_rec.processed_path):
            raise ValueError("No processed cardiac signal available for this assessment.")

        proc_signal = np.loadtxt(cardiac_rec.processed_path, dtype=np.float32)
        
        # Prepare normalized windowed sequences
        norm_sig = (proc_signal - 130.0) / 25.0
        window_len = Config.CARDIAC_WINDOW_LEN
        if len(norm_sig) < window_len:
            padded = np.pad(norm_sig, (0, window_len - len(norm_sig)), mode='edge')
            cardiac_tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(-1).float()
        else:
            # Use middle representative window
            start_idx = (len(norm_sig) - window_len) // 2
            window = norm_sig[start_idx : start_idx + window_len]
            cardiac_tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1).float()

        # 3. Perform Multi-Modal Forward Pass
        with torch.no_grad():
            f_brain = self.fusion_model.brain_cnn.extract_features(brain_tensor)
            f_cardiac = self.fusion_model.cardiac_lstm.extract_features(cardiac_tensor)

            # Feature-level concatenation and gating
            f_fused, gates = self.fusion_model.fuse_features(f_brain, f_cardiac)
            logits = self.fusion_model.fusion_mlp(f_fused)
            probabilities = F.softmax(logits, dim=-1).squeeze(0).numpy()

            # Continuous risk score index (0.0 to 1.0)
            if Config.NUM_CLASSES == 3:
                # Normal (weight 0.0), Suspect (weight 0.5), Pathological (weight 1.0)
                risk_score = float(0.0 * probabilities[0] + 0.5 * probabilities[1] + 1.0 * probabilities[2])
                classes = MultiModalFusionModel.CLASS_LABELS_3
            else:
                risk_score = float(probabilities[1])
                classes = MultiModalFusionModel.CLASS_LABELS_2

            pred_idx = int(np.argmax(probabilities))
            predicted_class = classes[pred_idx]
            confidence = float(probabilities[pred_idx])

            # Modality contribution analysis from gate activations
            gate_weights = gates.squeeze(0).numpy()
            brain_gate_mean = float(np.mean(gate_weights[:Config.BRAIN_FEATURE_DIM]))
            cardiac_gate_mean = float(np.mean(gate_weights[Config.BRAIN_FEATURE_DIM:]))
            total_gate = brain_gate_mean + cardiac_gate_mean + 1e-8
            brain_influence_pct = round((brain_gate_mean / total_gate) * 100.0, 1)
            cardiac_influence_pct = round((cardiac_gate_mean / total_gate) * 100.0, 1)

        # 4. Compile Fusion Details
        fusion_summary = {
            "fusion_strategy": "Feature-Level Concatenation with Adaptive Gated Attention",
            "brain_feature_dim": Config.BRAIN_FEATURE_DIM,
            "cardiac_feature_dim": Config.CARDIAC_FEATURE_DIM,
            "fused_feature_dim": Config.BRAIN_FEATURE_DIM + Config.CARDIAC_FEATURE_DIM,
            "brain_modality_influence_pct": brain_influence_pct,
            "cardiac_modality_influence_pct": cardiac_influence_pct,
            "class_probabilities": {
                classes[i]: round(float(probabilities[i]), 4) for i in range(len(classes))
            },
            "risk_index_percent": round(risk_score * 100.0, 2),
            "disclaimer": Config.DISCLAIMER,
        }

        # 5. Save Risk Assessment to Database
        assessment_uid = f"ASM_{uuid.uuid4().hex[:8].upper()}"
        assessment_rec = RiskAssessment(
            assessment_id=assessment_uid,
            patient_id=patient.id,
            signal_id=cardiac_rec.id,
            image_id=brain_rec.id,
            risk_score=risk_score,
            prediction=predicted_class,
            confidence=confidence,
            fusion_details=json.dumps(fusion_summary),
            clinician_notes=clinician_notes,
        )
        db.session.add(assessment_rec)
        db.session.commit()

        return {
            "assessment_id": assessment_uid,
            "db_id": assessment_rec.id,
            "patient_id": patient.patient_id,
            "patient_name": patient.name,
            "gestational_week": patient.gestational_week,
            "risk_score": round(risk_score, 4),
            "risk_percent": round(risk_score * 100.0, 1),
            "prediction": predicted_class,
            "confidence": round(confidence * 100.0, 1),
            "probabilities": fusion_summary["class_probabilities"],
            "modality_influence": {
                "brain_imaging": brain_influence_pct,
                "cardiac_signal": cardiac_influence_pct,
            },
            "fusion_details": fusion_summary,
            "cardiac_stats": json.loads(cardiac_rec.signal_stats) if cardiac_rec.signal_stats else {},
            "brain_metrics": json.loads(brain_rec.atlas_mapping) if brain_rec.atlas_mapping else {},
            "brain_images": {
                "preprocessed_url": f"/api/brain/image/{os.path.basename(brain_rec.preprocessed_path)}" if brain_rec.preprocessed_path else None,
                "atlas_mapped_url": f"/api/brain/image/{os.path.basename(brain_rec.atlas_mapped_path)}" if brain_rec.atlas_mapped_path else None,
                "deformation_field_url": f"/api/brain/image/{os.path.basename(brain_rec.deformation_field_path)}" if brain_rec.deformation_field_path else None,
            },
            "disclaimer": Config.DISCLAIMER,
        }
