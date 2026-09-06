import os
import sys
import uuid
import json
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.config import Config
    from backend.preprocessing.cardiac_preprocessing import CardiacPreprocessor
    from backend.models.lstm_model import FetalCardiacLSTM
    from backend.database.db import db
    from backend.database.models import CardiacSignal
except (ImportError, ValueError):
    from ..config import Config
    from ..preprocessing.cardiac_preprocessing import CardiacPreprocessor
    from ..models.lstm_model import FetalCardiacLSTM
    from ..database.db import db
    from ..database.models import CardiacSignal

class CardiacService:
    def __init__(self):
        self.preprocessor = CardiacPreprocessor(
            sampling_rate=Config.CARDIAC_SAMPLING_RATE,
            window_len=Config.CARDIAC_WINDOW_LEN
        )
        self.lstm_model = FetalCardiacLSTM(
            input_dim=1, feature_dim=Config.CARDIAC_FEATURE_DIM
        )
        self.lstm_model.eval()

    def process_cardiac_signal(self, file_path, patient_id=None):
        """
        Executes complete cardiac branch:
        Upload -> Denoise/Filter -> Statistics -> Sequence Windowing -> LSTM Features -> DB Record
        """
        sig_uid = f"CARD_{uuid.uuid4().hex[:8].upper()}"

        # 1. Load raw signal
        raw_signal = self.preprocessor.load_signal(file_path)

        # 2. Preprocess (filter extreme values, interpolate gaps, smooth spikes)
        processed_signal, stats = self.preprocessor.preprocess(raw_signal)

        # 3. Save processed signal to disk
        proc_filename = f"{sig_uid}_processed.dat"
        proc_path = os.path.join(Config.CARDIAC_UPLOAD_FOLDER, proc_filename)
        np.savetxt(proc_path, processed_signal, fmt="%.2f")

        # 4. Prepare normalized sequences for LSTM
        sequences = self.preprocessor.prepare_sequences(processed_signal)

        # 5. Extract temporal features via LSTM
        seq_tensor = torch.from_numpy(sequences).float()
        with torch.no_grad():
            f_cardiac_batch = self.lstm_model.extract_features(seq_tensor).numpy()
            # Mean-pool over all windows for the patient's global representation
            f_cardiac = np.mean(f_cardiac_batch, axis=0)

        feature_summary = {
            "dimension": len(f_cardiac),
            "mean": float(np.mean(f_cardiac)),
            "std": float(np.std(f_cardiac)),
            "l2_norm": float(np.linalg.norm(f_cardiac)),
            "sample_vector": [round(float(v), 4) for v in f_cardiac[:16]],
        }

        # 6. Generate lightweight visualization points for web chart
        chart_payload = self.preprocessor.get_visualization_payload(raw_signal, processed_signal)

        # 7. Create database record if patient_id provided
        db_record = None
        if patient_id is not None:
            db_record = CardiacSignal(
                signal_id=sig_uid,
                patient_id=patient_id,
                heart_rate=stats["baseline_fhr"],
                variability=stats["short_term_variability"],
                raw_path=str(file_path),
                processed_path=str(proc_path),
                signal_stats=json.dumps({
                    "clinical_stats": stats,
                    "features": feature_summary,
                })
            )
            db.session.add(db_record)
            db.session.commit()

        return {
            "signal_id": sig_uid,
            "db_id": db_record.id if db_record else None,
            "patient_id": patient_id,
            "raw_filename": os.path.basename(file_path),
            "processed_filename": proc_filename,
            "stats": stats,
            "chart_data": chart_payload,
            "feature_summary": feature_summary,
            "status": "Denoised & Temporal LSTM Features Extracted",
        }
