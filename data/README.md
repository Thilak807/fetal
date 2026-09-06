# Multi-Modal Fetal Risk Assessment Datasets

## 1. Ethical & Regulatory Notice
> **IMPORTANT: DEMO DATA — NOT FOR CLINICAL USE**
> All sample data files provided in this repository are intended solely for testing, software validation, and research prototyping. They are NOT certified for clinical diagnosis, patient management, or medical decision-making.

---

## 2. Modality 1: Fetal Cardiac Signals
- **Primary Source**: PhysioNet CTU-CHB Intrapartum Cardiotocography Database (`CTU-CHB/dat/`).
- **Format**: Time-series `.dat`, `.csv`, or `.txt` sampled at 4 Hz.
- **Physical Meaning**: Fetal Heart Rate (FHR in beats per minute, bpm) and uterine contraction pressure (mmHg).
- **Normal Range**: 110–160 bpm with physiological short-term variability (STV > 5 bpm).

---

## 3. Modality 2: Fetal Brain Imaging
- **Format**: Grayscale `.png`, `.jpg`, `.tiff`, or NIfTI `.nii`/`.nii.gz`.
- **Target Anatomy**: Axial view of the fetal cranium (24–36 gestational weeks) displaying cerebral hemispheres, lateral ventricles, and cavum septi pellucidi (CSP).
- **Processing**: Preprocessed and mapped via **Elastic Deformable Atlas Registration** onto the standard normative fetal brain reference template before CNN feature extraction.

---

## 4. Multi-Modal Fusion
Both learned feature representations:
- $\mathbf{F}_{\text{brain}} \in \mathbb{R}^{128}$ (Spatial features from Brain CNN)
- $\mathbf{F}_{\text{cardiac}} \in \mathbb{R}^{128}$ (Temporal dynamics from Cardiac LSTM)

are combined at the feature level via concatenation with adaptive gated attention:
$$\mathbf{F}_{\text{fusion}} = [\mathbf{F}_{\text{brain}} \,\|\, \mathbf{F}_{\text{cardiac}}] \odot \mathbf{g}$$
to yield the final risk prediction and continuous risk index.
