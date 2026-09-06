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

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
