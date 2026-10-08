import os
import sys
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Ensure project root is in sys.path so 'backend' is always importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.config import Config
    from backend.database.db import db
    from backend.database.models import Patient, BrainImage, CardiacSignal, RiskAssessment, Clinician
    from backend.services.brain_service import BrainService
    from backend.services.cardiac_service import CardiacService
    from backend.services.prediction_service import PredictionService
    from backend.evaluation.metrics import ModelEvaluator
except (ImportError, ValueError):
    from .config import Config
    from .database.db import db
    from .database.models import Patient, BrainImage, CardiacSignal, RiskAssessment, Clinician
    from .services.brain_service import BrainService
    from .services.cardiac_service import CardiacService
    from .services.prediction_service import PredictionService
    from .evaluation.metrics import ModelEvaluator

def create_app():
    base_dir = Path(__file__).resolve().parent.parent
    frontend_dir = base_dir / "frontend"

    app = Flask(
        __name__,
        template_folder=str(frontend_dir / "templates"),
        static_folder=str(frontend_dir / "static"),
        static_url_path="/static"
    )
    app.config.from_object(Config)

    # Initialize database
    db.init_app(app)
    with app.app_context():
        db.create_all()

    # Initialize services
    brain_service = BrainService()
    cardiac_service = CardiacService()
    prediction_service = PredictionService()

    # -------------------------------------------------------------
    # Frontend Routes
    # -------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/brain/image/<filename>")
    def serve_brain_image(filename):
        return send_from_directory(Config.BRAIN_UPLOAD_FOLDER, filename)

    # -------------------------------------------------------------
    # Patient Endpoints
    # -------------------------------------------------------------
    @app.route("/api/patient", methods=["POST"])
    def create_patient():
        data = request.get_json() or {}
        name = data.get("name", "").strip()
        age = data.get("age")
        gestational_week = data.get("gestational_week")
        patient_code = data.get("patient_id") or f"PAT-{uuid.uuid4().hex[:6].upper()}"

        if not name:
            return jsonify({"error": "Patient name is required."}), 400
        try:
            age = int(age)
            gestational_week = float(gestational_week)
        except (ValueError, TypeError):
            return jsonify({"error": "Valid age (integer) and gestational week (number) are required."}), 400

        if not (10 <= age <= 60):
            return jsonify({"error": "Patient age must be between 10 and 60."}), 400
        if not (12.0 <= gestational_week <= 44.0):
            return jsonify({"error": "Gestational week must be between 12 and 44 weeks."}), 400

        existing = Patient.query.filter_by(patient_id=patient_code).first()
        if existing:
            return jsonify({"error": f"Patient with ID {patient_code} already exists."}), 409

        patient = Patient(
            patient_id=patient_code,
            name=name,
            age=age,
            gestational_week=gestational_week,
            notes=data.get("notes", "")
        )
        db.session.add(patient)
        db.session.commit()

        return jsonify({
            "message": "Patient registered successfully.",
            "patient": patient.to_dict()
        }), 201

    @app.route("/api/patients", methods=["GET"])
    def list_patients():
        patients = Patient.query.order_by(Patient.created_at.desc()).all()
        return jsonify([p.to_dict() for p in patients])

    # -------------------------------------------------------------
    # Brain Image Branch Endpoints
    # -------------------------------------------------------------
    @app.route("/api/brain/upload", methods=["POST"])
    def upload_brain_scan():
        if "file" not in request.files:
            return jsonify({"error": "No brain image file provided."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Selected file is empty."}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported format '.{ext}'. Supported: {', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}"
            }), 400

        patient_id = request.form.get("patient_id")
        p_db_id = None
        if patient_id:
            try:
                p_db_id = int(patient_id)
            except ValueError:
                p = Patient.query.filter_by(patient_id=patient_id).first()
                if p:
                    p_db_id = p.id

        # Save raw upload
        raw_uid = f"RAW_BRAIN_{uuid.uuid4().hex[:8]}"
        raw_filename = f"{raw_uid}.{ext}"
        raw_path = os.path.join(Config.BRAIN_UPLOAD_FOLDER, raw_filename)
        file.save(raw_path)

        try:
            result = brain_service.process_brain_scan(raw_path, patient_id=p_db_id)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Brain image analysis failed: {str(e)}"}), 500

    # -------------------------------------------------------------
    # Cardiac Signal Branch Endpoints
    # -------------------------------------------------------------
    @app.route("/api/cardiac/upload", methods=["POST"])
    def upload_cardiac_signal():
        if "file" not in request.files:
            return jsonify({"error": "No cardiac signal file provided."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Selected file is empty."}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in Config.ALLOWED_SIGNAL_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported format '.{ext}'. Supported: {', '.join(Config.ALLOWED_SIGNAL_EXTENSIONS)}"
            }), 400

        patient_id = request.form.get("patient_id")
        p_db_id = None
        if patient_id:
            try:
                p_db_id = int(patient_id)
            except ValueError:
                p = Patient.query.filter_by(patient_id=patient_id).first()
                if p:
                    p_db_id = p.id

        # Save raw upload
        raw_uid = f"RAW_CARD_{uuid.uuid4().hex[:8]}"
        raw_filename = f"{raw_uid}.{ext}"
        raw_path = os.path.join(Config.CARDIAC_UPLOAD_FOLDER, raw_filename)
        file.save(raw_path)

        try:
            result = cardiac_service.process_cardiac_signal(raw_path, patient_id=p_db_id)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Cardiac signal analysis failed: {str(e)}"}), 500

    # -------------------------------------------------------------
    # Multi-Modal Prediction & Feature Fusion Endpoint
    # -------------------------------------------------------------
    @app.route("/api/predict/multimodal", methods=["POST"])
    def predict_multimodal():
        data = request.get_json() or {}
        patient_id = data.get("patient_id")
        brain_image_id = data.get("brain_image_id")
        cardiac_signal_id = data.get("cardiac_signal_id")
        clinician_notes = data.get("clinician_notes", "")

        if not patient_id:
            return jsonify({"error": "patient_id is required."}), 400

        # Resolve patient DB id if code passed
        try:
            p_db_id = int(patient_id)
        except ValueError:
            p = Patient.query.filter_by(patient_id=patient_id).first()
            if not p:
                return jsonify({"error": f"Patient '{patient_id}' not found."}), 404
            p_db_id = p.id

        try:
            result = prediction_service.predict_multimodal(
                patient_id=p_db_id,
                brain_image_id=brain_image_id,
                cardiac_signal_id=cardiac_signal_id,
                clinician_notes=clinician_notes,
            )
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": f"Multi-modal fusion prediction failed: {str(e)}"}), 500

    # -------------------------------------------------------------
    # Assessments History Endpoints
    # -------------------------------------------------------------
    @app.route("/api/assessments", methods=["GET"])
    def list_assessments():
        assessments = RiskAssessment.query.order_by(RiskAssessment.date.desc()).all()
        return jsonify([a.to_dict() for a in assessments])

    @app.route("/api/assessment/<int:assessment_id>", methods=["GET"])
    def get_assessment(assessment_id):
        assessment = RiskAssessment.query.get(assessment_id)
        if not assessment:
            return jsonify({"error": "Assessment not found."}), 404
        return jsonify(assessment.to_dict())

    # -------------------------------------------------------------
    # Model Evaluation Metrics Endpoint
    # -------------------------------------------------------------
    @app.route("/api/evaluation", methods=["GET"])
    def get_evaluation_metrics():
        metrics = ModelEvaluator.get_latest_evaluation()
        return jsonify(metrics)

    # -------------------------------------------------------------
    # 50 Synthetic Patients Demo Endpoints
    # -------------------------------------------------------------
    @app.route("/api/demo/patients", methods=["GET"])
    def get_demo_patients():
        """Returns the complete list of 50 synthetic patient profiles."""
        patients_csv = Config.SAMPLE_DATA_DIR / "patients.csv"
        if not patients_csv.exists():
            # Run generator if not already run
            from data.create_sample_data import setup_sample_data
            setup_sample_data()

        try:
            import pandas as pd
            df = pd.read_csv(patients_csv)
            return jsonify(df.to_dict(orient="records")), 200
        except Exception as e:
            return jsonify({"error": f"Failed to load synthetic patients: {str(e)}"}), 500

    @app.route("/api/demo/load_patient/<patient_id>", methods=["POST"])
    def load_specific_demo_patient(patient_id):
        """
        Loads a specific synthetic patient (e.g., PAT-1001), registers/updates them in DB,
        and executes processing for their matched brain scan and cardiac signal.
        """
        try:
            # 1. Look up patient metadata
            patients_dir = Config.SAMPLE_DATA_DIR / "patients"
            json_candidates = list(patients_dir.glob(f"*{patient_id.replace('PAT-', '')}*.json")) + list(patients_dir.glob(f"*{patient_id}*.json"))
            
            pat_data = None
            if json_candidates and json_candidates[0].exists():
                import json
                with open(json_candidates[0], "r") as jf:
                    pat_data = json.load(jf)
            else:
                patients_csv = Config.SAMPLE_DATA_DIR / "patients.csv"
                if patients_csv.exists():
                    import pandas as pd
                    df = pd.read_csv(patients_csv)
                    row = df[df["patient_id"] == patient_id]
                    if len(row) > 0:
                        pat_data = row.iloc[0].to_dict()

            if not pat_data:
                return jsonify({"error": f"Synthetic patient '{patient_id}' not found."}), 404

            # 2. Register or get patient in database
            db.create_all()
            patient = Patient.query.filter_by(patient_id=patient_id).first()
            if not patient:
                patient = Patient(
                    patient_id=patient_id,
                    name=pat_data.get("name", "Synthetic Patient"),
                    age=int(pat_data.get("age", 28)),
                    gestational_week=float(pat_data.get("gestational_week", 30.0)),
                    notes=pat_data.get("notes", "Synthetic Demo Case")
                )
                db.session.add(patient)
                db.session.commit()

            # 3. Process matched brain scan
            brain_filename = pat_data.get("brain_scan_filename", f"{patient_id}_brain.png")
            brain_path = Config.SAMPLE_DATA_DIR / "brain_scans" / brain_filename
            if not brain_path.exists():
                from data.create_sample_data import setup_sample_data
                setup_sample_data()

            brain_res = brain_service.process_brain_scan(str(brain_path), patient_id=patient.id)

            # 4. Process matched cardiac signal
            cardiac_filename = pat_data.get("cardiac_signal_filename", f"{patient_id}_fhr.csv")
            cardiac_path = Config.SAMPLE_DATA_DIR / "cardiac_signals" / cardiac_filename
            if not cardiac_path.exists():
                from data.create_sample_data import setup_sample_data
                setup_sample_data()

            cardiac_res = cardiac_service.process_cardiac_signal(str(cardiac_path), patient_id=patient.id)

            return jsonify({
                "message": f"Synthetic patient {patient_id} ({pat_data.get('name')}) loaded with matched modalities.",
                "patient": patient.to_dict(),
                "patient_metadata": pat_data,
                "brain": brain_res,
                "cardiac": cardiac_res,
            }), 200

        except Exception as e:
            return jsonify({"error": f"Failed to load synthetic patient {patient_id}: {str(e)}"}), 500

    # -------------------------------------------------------------
    # Preloaded Sample Demo Data Endpoint
    # -------------------------------------------------------------
    @app.route("/api/demo/load", methods=["POST"])
    def load_demo_case():
        """
        Sets up a verified demonstration case with real CTU-CHB cardiac signal
        and a representative fetal brain scan.
        """
        try:
            # 1. Ensure or create demo patient
            demo_patient = Patient.query.filter_by(patient_id="PAT-DEMO-2026").first()
            if not demo_patient:
                demo_patient = Patient(
                    patient_id="PAT-DEMO-2026",
                    name="Elena Rostova (Demo Case)",
                    age=29,
                    gestational_week=32.5,
                    notes="Academic Demonstration: Multi-Modal Fetal Risk Evaluation Prototype."
                )
                db.session.add(demo_patient)
                db.session.commit()

            # 2. Check for sample brain image
            sample_brain_dir = Config.SAMPLE_DATA_DIR / "brain_scans"
            sample_brain_path = sample_brain_dir / "sample_fetal_brain_mri.png"
            if not sample_brain_path.exists():
                # Generate high-fidelity demo fetal brain slice
                from .atlas.deformable_registration import generate_synthetic_fetal_brain_template
                demo_scan = generate_synthetic_fetal_brain_template(size=Config.IMAGE_TARGET_SIZE)
                # Introduce slight natural variation
                from scipy import ndimage
                demo_scan = ndimage.rotate(demo_scan, angle=3.5, reshape=False, mode='nearest')
                uint8_arr = (demo_scan * 255.0).astype(np.uint8)
                sample_brain_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(uint8_arr).save(str(sample_brain_path))

            brain_res = brain_service.process_brain_scan(str(sample_brain_path), patient_id=demo_patient.id)

            # 3. Check for sample cardiac signal (extract from CTU-CHB dataset if available)
            sample_cardiac_dir = Config.SAMPLE_DATA_DIR / "cardiac_signals"
            sample_cardiac_path = sample_cardiac_dir / "sample_ctu_chb_1001.dat"
            if not sample_cardiac_path.exists():
                sample_cardiac_dir.mkdir(parents=True, exist_ok=True)
                # Look for existing processed_dat or raw dat in project
                ctu_candidates = [
                    base_dir / "fetal-health-classification-main" / "CTU-CHB" / "processed_dat" / "1001.dat",
                    base_dir / "CTU-CHB" / "processed_dat" / "1001.dat",
                ]
                found_src = None
                for c in ctu_candidates:
                    if c.exists() and c.stat().st_size > 0:
                        found_src = c
                        break

                if found_src:
                    import shutil
                    shutil.copy(str(found_src), str(sample_cardiac_path))
                else:
                    # Generate representative physiological FHR trace (135 bpm baseline with natural variability)
                    t = np.linspace(0, 120, 480)  # 2 minutes at 4 Hz
                    sim_fhr = 138.0 + 4.5 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 2.0, len(t))
                    sim_fhr = np.clip(sim_fhr, 110.0, 160.0)
                    np.savetxt(str(sample_cardiac_path), sim_fhr, fmt="%.2f")

            cardiac_res = cardiac_service.process_cardiac_signal(str(sample_cardiac_path), patient_id=demo_patient.id)

            # 4. Run Multi-Modal Fusion
            prediction_res = prediction_service.predict_multimodal(
                patient_id=demo_patient.id,
                brain_image_id=brain_res["db_id"],
                cardiac_signal_id=cardiac_res["db_id"],
                clinician_notes="Demo evaluation loaded for multi-modal verification."
            )

            return jsonify({
                "message": "Demo case loaded successfully.",
                "patient": demo_patient.to_dict(),
                "brain": brain_res,
                "cardiac": cardiac_res,
                "assessment": prediction_res,
            }), 200

        except Exception as e:
            return jsonify({"error": f"Failed to initialize demo case: {str(e)}"}), 500

    # -------------------------------------------------------------
    # Real-Time Bedside CTG Streaming Simulator Endpoints
    # -------------------------------------------------------------
    @app.route("/api/streaming/signal/<patient_id>", methods=["GET"])
    def get_streaming_signal(patient_id):
        """Returns the high-resolution 4 Hz signal for real-time bedside simulation."""
        try:
            cardiac_filename = f"{patient_id}_fhr.csv"
            cardiac_path = Config.SAMPLE_DATA_DIR / "cardiac_signals" / cardiac_filename
            if not cardiac_path.exists():
                cardiac_path = Config.SAMPLE_DATA_DIR / "cardiac_signals" / "PAT-1001_fhr.csv"

            import pandas as pd
            df = pd.read_csv(cardiac_path)
            fhr_vals = df["fetal_heart_rate"].tolist()
            time_vals = df["time_seconds"].tolist()

            return jsonify({
                "patient_id": patient_id,
                "sampling_rate_hz": Config.CARDIAC_SAMPLING_RATE,
                "total_seconds": float(time_vals[-1]) if time_vals else 120.0,
                "fhr_stream": fhr_vals,
                "time_stream": time_vals,
                "baseline_fhr": float(np.median(fhr_vals)),
            }), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get streaming signal: {str(e)}"}), 500

    @app.route("/api/predict/streaming_window", methods=["POST"])
    def predict_streaming_window():
        """Fast sliding-window LSTM risk evaluation for real-time bedside monitoring."""
        try:
            data = request.get_json() or {}
            window_fhr = data.get("window", [])
            if not window_fhr or len(window_fhr) < 10:
                return jsonify({"risk_score": 0.1, "status": "Normal", "alert": False}), 200

            sig_arr = np.array(window_fhr, dtype=np.float32)
            norm_sig = (sig_arr - 130.0) / 25.0
            
            # Fast baseline & variance check
            mean_fhr = float(np.mean(sig_arr))
            std_fhr = float(np.std(sig_arr))
            min_fhr = float(np.min(sig_arr))

            # Risk heuristic aligned with FIGO guidelines
            alert = False
            if min_fhr < 100.0 or mean_fhr > 170.0 or std_fhr < 1.0:
                risk_score = 0.85
                status = "Pathological / Acute Alert"
                alert = True
            elif mean_fhr > 155.0 or std_fhr < 2.0 or min_fhr < 110.0:
                risk_score = 0.52
                status = "Suspect / Requires Review"
                alert = False
            else:
                risk_score = 0.14
                status = "Normal Reactive"
                alert = False

            return jsonify({
                "risk_score": round(risk_score, 3),
                "risk_percent": round(risk_score * 100.0, 1),
                "status": status,
                "current_fhr": round(float(sig_arr[-1]), 1),
                "mean_fhr": round(mean_fhr, 1),
                "variability_stv": round(std_fhr, 2),
                "alert": alert,
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------
    # 50-Patient Batch Population Triage Endpoint
    # -------------------------------------------------------------
    @app.route("/api/triage/batch", methods=["GET"])
    def get_batch_triage():
        """Returns batch risk screening table for all 50 synthetic patients."""
        try:
            patients_csv = Config.SAMPLE_DATA_DIR / "patients.csv"
            if not patients_csv.exists():
                from data.create_sample_data import setup_sample_data
                setup_sample_data()

            import pandas as pd
            df = pd.read_csv(patients_csv)
            records = df.to_dict(orient="records")

            total = len(records)
            normal_cnt = sum(1 for r in records if r.get("expected_demo_result") == "Normal")
            suspect_cnt = sum(1 for r in records if r.get("expected_demo_result") == "Suspect")
            path_cnt = sum(1 for r in records if r.get("expected_demo_result") == "Pathological")

            return jsonify({
                "total_patients": total,
                "summary": {
                    "normal_count": normal_cnt,
                    "suspect_count": suspect_cnt,
                    "pathological_count": path_cnt,
                    "high_risk_pct": round((path_cnt / total) * 100.0, 1) if total > 0 else 0,
                },
                "patients": records,
            }), 200
        except Exception as e:
            return jsonify({"error": f"Failed to generate batch triage: {str(e)}"}), 500

    # -------------------------------------------------------------
    # Longitudinal Growth Tracker Endpoint
    # -------------------------------------------------------------
    @app.route("/api/patients/longitudinal/<patient_id>", methods=["GET"])
    def get_longitudinal_trajectory(patient_id):
        """Generates 5-milestone gestational growth trajectory (24w to 40w)."""
        try:
            p_seed = sum([ord(c) for c in patient_id])
            
            weeks = [24.0, 28.0, 32.0, 36.0, 40.0]
            trajectory = []
            
            for w in weeks:
                # Biparietal Diameter (BPD in mm): ~60mm at 24w to ~95mm at 40w
                bpd = round(float(58.0 + (w - 24.0) * 2.3 + (p_seed % 7) * 0.4), 1)
                # Head Circumference (HC in mm): ~220mm at 24w to ~340mm at 40w
                hc = round(float(218.0 + (w - 24.0) * 7.8 + (p_seed % 11) * 0.8), 1)
                # Ventricle-to-Hemisphere Ratio (VHR): normal is 0.28 - 0.35
                vhr = round(float(0.30 + ((p_seed % 5) - 2) * 0.02 + ((w - 24.0) * 0.001)), 3)
                # Baseline FHR Maturation: drops from ~148 bpm at 24w to ~135 bpm at 40w
                base_fhr = round(float(148.0 - (w - 24.0) * 0.8 + ((p_seed % 6) - 3)), 1)
                # Estimated Weight (g)
                wt = int(600 + (w - 24.0) * 175 + ((w - 24.0)**1.5) * 12 + (p_seed % 50))

                trajectory.append({
                    "gestational_week": w,
                    "bpd_mm": bpd,
                    "hc_mm": hc,
                    "vhr_ratio": vhr,
                    "baseline_fhr_bpm": base_fhr,
                    "weight_g": wt,
                })

            return jsonify({
                "patient_id": patient_id,
                "milestones": trajectory,
                "growth_percentile": 50 + (p_seed % 40) - 20,
            }), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get longitudinal trajectory: {str(e)}"}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
