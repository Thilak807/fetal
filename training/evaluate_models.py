import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from backend.config import Config
from backend.models.fusion_model import MultiModalFusionModel
from backend.evaluation.metrics import ModelEvaluator

def evaluate_on_test_split():
    print("==================================================")
    print("Multi-Modal Fetal Risk Model Genuine Test Evaluation")
    print("==================================================")

    test_split_path = Config.SAVED_MODELS_DIR / "test_split.npz"
    weights_path = Config.SAVED_MODELS_DIR / "multimodal_fusion.pt"

    if not test_split_path.exists():
        print("[ERROR] No test split found. Please run train_multimodal.py first.")
        return

    if not weights_path.exists():
        print("[ERROR] No trained weights found. Please run train_multimodal.py first.")
        return

    # Load test split
    data = np.load(str(test_split_path))
    X_brain = data["brain"]
    X_cardiac = data["cardiac"]
    y_true = data["labels"]

    print(f"Loaded {len(y_true)} held-out test cases.")

    # Load Model
    model = MultiModalFusionModel(
        brain_feature_dim=Config.BRAIN_FEATURE_DIM,
        cardiac_feature_dim=Config.CARDIAC_FEATURE_DIM,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        use_gated_attention=True
    )
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()

    b_tensor = torch.from_numpy(X_brain).unsqueeze(1).float()
    c_tensor = torch.from_numpy(X_cardiac).float()

    with torch.no_grad():
        logits, _, _, _ = model(b_tensor, c_tensor)
        probabilities = F.softmax(logits, dim=-1).numpy()
        y_pred = np.argmax(probabilities, axis=-1)

    classes = MultiModalFusionModel.CLASS_LABELS_3
    metrics = ModelEvaluator.compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=probabilities,
        class_names=classes
    )

    print("\n--- Genuine Evaluation Results ---")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

    # Save results to JSON
    out_file = Config.SAVED_MODELS_DIR / "evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation metrics successfully written to: {out_file}")
    return metrics

if __name__ == "__main__":
    evaluate_on_test_split()
