from datetime import datetime
import json
from .db import db

class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(128), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gestational_week = db.Column(db.Float, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    cardiac_signals = db.relationship("CardiacSignal", backref="patient", lazy=True, cascade="all, delete-orphan")
    brain_images = db.relationship("BrainImage", backref="patient", lazy=True, cascade="all, delete-orphan")
    assessments = db.relationship("RiskAssessment", backref="patient", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gestational_week": self.gestational_week,
            "notes": self.notes,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "assessment_count": len(self.assessments),
        }


class CardiacSignal(db.Model):
    __tablename__ = "cardiac_signals"

    id = db.Column(db.Integer, primary_key=True)
    signal_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    heart_rate = db.Column(db.Float, nullable=True)  # Mean baseline FHR (bpm)
    variability = db.Column(db.Float, nullable=True)  # Short-term / Long-term variability
    raw_path = db.Column(db.String(256), nullable=False)
    processed_path = db.Column(db.String(256), nullable=True)
    signal_stats = db.Column(db.Text, nullable=True)  # JSON-encoded statistics
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    assessments = db.relationship("RiskAssessment", backref="cardiac_signal", lazy=True)

    def to_dict(self):
        stats = {}
        if self.signal_stats:
            try:
                stats = json.loads(self.signal_stats)
            except Exception:
                pass
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None,
            "heart_rate": self.heart_rate,
            "variability": self.variability,
            "raw_path": self.raw_path,
            "processed_path": self.processed_path,
            "signal_stats": stats,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }


class BrainImage(db.Model):
    __tablename__ = "brain_images"

    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    raw_path = db.Column(db.String(256), nullable=False)
    preprocessed_path = db.Column(db.String(256), nullable=True)
    atlas_mapped_path = db.Column(db.String(256), nullable=True)
    deformation_field_path = db.Column(db.String(256), nullable=True)
    atlas_mapping = db.Column(db.Text, nullable=True)  # JSON-encoded registration metrics
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    assessments = db.relationship("RiskAssessment", backref="brain_image", lazy=True)

    def to_dict(self):
        metrics = {}
        if self.atlas_mapping:
            try:
                metrics = json.loads(self.atlas_mapping)
            except Exception:
                pass
        return {
            "id": self.id,
            "image_id": self.image_id,
            "patient_id": self.patient_id,
            "scan_date": self.scan_date.strftime("%Y-%m-%d %H:%M:%S") if self.scan_date else None,
            "raw_path": self.raw_path,
            "preprocessed_path": self.preprocessed_path,
            "atlas_mapped_path": self.atlas_mapped_path,
            "deformation_field_path": self.deformation_field_path,
            "atlas_mapping": metrics,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }


class RiskAssessment(db.Model):
    __tablename__ = "risk_assessments"

    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    signal_id = db.Column(db.Integer, db.ForeignKey("cardiac_signals.id"), nullable=True)
    image_id = db.Column(db.Integer, db.ForeignKey("brain_images.id"), nullable=True)
    risk_score = db.Column(db.Float, nullable=False)  # Normalized 0.0 to 1.0 (or percentage 0 - 100%)
    prediction = db.Column(db.String(64), nullable=False)  # Normal, Suspect, Pathological (or At-Risk)
    confidence = db.Column(db.Float, nullable=False)  # Model confidence 0.0 to 1.0
    date = db.Column(db.DateTime, default=datetime.utcnow)
    fusion_details = db.Column(db.Text, nullable=True)  # JSON-encoded fusion weights and feature stats
    clinician_notes = db.Column(db.Text, nullable=True)

    def to_dict(self):
        details = {}
        if self.fusion_details:
            try:
                details = json.loads(self.fusion_details)
            except Exception:
                pass
        return {
            "id": self.id,
            "assessment_id": self.assessment_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient.name if self.patient else None,
            "patient_code": self.patient.patient_id if self.patient else None,
            "gestational_week": self.patient.gestational_week if self.patient else None,
            "signal_id": self.signal_id,
            "image_id": self.image_id,
            "risk_score": round(self.risk_score, 4),
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "date": self.date.strftime("%Y-%m-%d %H:%M:%S") if self.date else None,
            "fusion_details": details,
            "clinician_notes": self.clinician_notes,
        }


class Clinician(db.Model):
    __tablename__ = "clinicians"

    id = db.Column(db.Integer, primary_key=True)
    clinician_id = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    department = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(128), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "clinician_id": self.clinician_id,
            "name": self.name,
            "department": self.department,
            "email": self.email,
        }
