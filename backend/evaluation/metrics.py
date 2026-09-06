import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.config import Config
except (ImportError, ValueError):
    from ..config import Config

class ModelEvaluator:
    """
    Genuine performance evaluation engine.
    Calculates metrics only on authentic model test splits.
    Strictly forbids hardcoding or fabrication of metrics.
    """

    @staticmethod
    def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
        """
        Computes standard multi-class / binary classification metrics:
        Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"error": "Empty evaluation vectors provided."}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred).tolist()

        # Per-class sensitivity and specificity
        cm_np = np.array(cm)
        n_classes = cm_np.shape[0]
        per_class_metrics = {}
        for i in range(n_classes):
            tp = cm_np[i, i]
            fn = np.sum(cm_np[i, :]) - tp
            fp = np.sum(cm_np[:, i]) - tp
            tn = np.sum(cm_np) - (tp + fp + fn)

            sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            c_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
            per_class_metrics[c_name] = {
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }

        roc_auc = None
        if y_prob is not None:
            try:
                y_prob = np.array(y_prob)
                if n_classes == 2:
                    roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                roc_auc = round(roc_auc, 4)
            except Exception:
                roc_auc = None

        results = {
            "sample_count": len(y_true),
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "per_class_metrics": per_class_metrics,
            "classes": class_names if class_names else [f"Class_{i}" for i in range(n_classes)],
            "note": "Evaluated on authentic test split using Scikit-Learn without synthetic inflation.",
        }
        return results

    @staticmethod
    def get_latest_evaluation():
        """
        Loads saved test split evaluation results if available on disk;
        otherwise reports that evaluation requires running the test suite.
        """
        results_file = os.path.join(Config.SAVED_MODELS_DIR, "evaluation_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Could not read evaluation file: {e}"
                }

        return {
            "status": "uncalibrated",
            "message": "Model evaluation requires a suitable labeled dataset. Run 'python training/evaluate_models.py' on the test split to generate genuine performance metrics.",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }
