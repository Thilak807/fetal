import sys
import os
import json
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from backend.config import Config
from backend.atlas.deformable_registration import (
    generate_synthetic_fetal_brain_template,
    get_default_atlas_template,
)

# 50 Fictional Patient Names (25 female / 25 male maternal aliases for synthetic demo)
FICTIONAL_NAMES = [
    "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", "David Wilson",
    "Jessica Taylor", "James Anderson", "Ashley Thomas", "Robert Jackson", "Amanda White",
    "William Harris", "Stephanie Martin", "Joseph Thompson", "Melissa Garcia", "Charles Martinez",
    "Nicole Robinson", "Thomas Clark", "Elizabeth Rodriguez", "Daniel Lewis", "Heather Lee",
    "Matthew Walker", "Amber Hall", "Anthony Allen", "Megan Young", "Donald Hernandez",
    "Rachel King", "Mark Wright", "Lauren Lopez", "Paul Hill", "Kayla Scott",
    "Andrew Green", "Victoria Adams", "Joshua Baker", "Brittany Gonzalez", "Kenneth Nelson",
    "Samantha Carter", "Kevin Mitchell", "Danielle Perez", "Brian Roberts", "Hannah Turner",
    "George Phillips", "Courtney Campbell", "Edward Parker", "Rebecca Evans", "Ronald Edwards",
    "Vanessa Collins", "Timothy Stewart", "Tiffany Sanchez", "Jason Morris", "Katherine Rogers"
]

def generate_synthetic_brain_scan(gestational_week, pattern_type="normal", scan_type="MRI", seed=1001):
    """
    Generates a 128x128 synthetic fetal brain scan image with realistic anatomical characteristics.
    """
    rng = np.random.RandomState(seed)
    base_template = generate_synthetic_fetal_brain_template(size=Config.IMAGE_TARGET_SIZE)
    h, w = Config.IMAGE_TARGET_SIZE
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0

    # 1. Gestational scaling (head size grows from 24 to 40 weeks)
    scale_factor = 0.82 + (gestational_week - 24.0) * (0.36 / 16.0)
    scaled_template = ndimage.zoom(base_template, scale_factor, order=1)
    
    # Crop or pad to 128x128
    img = np.zeros(Config.IMAGE_TARGET_SIZE, dtype=np.float32)
    sh, sw = scaled_template.shape
    r_start = max(0, (h - sh) // 2)
    c_start = max(0, (w - sw) // 2)
    sr_start = max(0, (sh - h) // 2)
    sc_start = max(0, (sw - w) // 2)
    
    copy_h = min(h, sh)
    copy_w = min(w, sw)
    img[r_start:r_start+copy_h, c_start:c_start+copy_w] = scaled_template[sr_start:sr_start+copy_h, sc_start:sc_start+copy_w]

    # 2. Pattern variations (Normal vs Variant vs Suspect/Pathological)
    if pattern_type == "mild_variant":
        # Slight rotation + mild ventricle dilation
        angle = rng.uniform(-4.5, 4.5)
        img = ndimage.rotate(img, angle=angle, reshape=False, mode='nearest')
        ventricle_mask = (img > 0.1) & (img < 0.25)
        img[ventricle_mask] *= 0.85
    elif pattern_type in ["suspect", "pathological"]:
        # Asymmetrical ventricle enlargement (Ventriculomegaly prototype)
        angle = rng.uniform(-6.0, 6.0)
        img = ndimage.rotate(img, angle=angle, reshape=False, mode='nearest')
        dist_from_c = np.sqrt((x - cx)**2 + (y - cy)**2)
        vent_region = (dist_from_c < 28) & (img > 0.1)
        img[vent_region] = 0.12
        cortex_mask = (dist_from_c >= 40) & (dist_from_c <= 54)
        img[cortex_mask] *= 0.85
    else: # normal
        angle = rng.uniform(-2.0, 2.0)
        if abs(angle) > 0.5:
            img = ndimage.rotate(img, angle=angle, reshape=False, mode='nearest')

    # 3. Modality texture (MRI vs Ultrasound)
    if scan_type == "Ultrasound":
        speckle = rng.normal(1.0, 0.12, img.shape)
        img = img * speckle
        shadow = np.exp(-((y - h)**2) / (2 * (h * 0.7)**2))
        img = img * (0.7 + 0.3 * shadow)
    else: # MRI
        noise = rng.normal(0.0, 0.02, img.shape)
        img = img + noise

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def generate_synthetic_cardiac_signal(gestational_week, pattern_type="normal", duration_sec=120, sampling_rate=4, seed=1001):
    """
    Generates a realistic-looking synthetic FHR time series (at 4 Hz) with physiological dynamics.
    """
    rng = np.random.RandomState(seed)
    num_points = int(duration_sec * sampling_rate)
    time_arr = np.linspace(0, duration_sec, num_points)

    if pattern_type == "normal":
        baseline = rng.uniform(130.0, 145.0)
        stv = rng.uniform(2.5, 4.5)
        fhr = baseline + stv * np.sin(2 * np.pi * 0.04 * time_arr) + rng.normal(0, 1.8, num_points)
        num_acc = rng.randint(2, 4)
        for _ in range(num_acc):
            center_t = rng.uniform(15, duration_sec - 15)
            width_t = rng.uniform(12, 22)
            fhr += 18.0 * np.exp(-((time_arr - center_t) ** 2) / (2 * (width_t / 2.5) ** 2))

    elif pattern_type == "mild_variant" or pattern_type == "suspect":
        baseline = rng.uniform(155.0, 168.0)
        stv = rng.uniform(1.2, 2.2)
        fhr = baseline + stv * np.sin(2 * np.pi * 0.03 * time_arr) + rng.normal(0, 1.2, num_points)
        for _ in range(2):
            center_t = rng.uniform(25, duration_sec - 20)
            fhr -= 16.0 * np.exp(-((time_arr - center_t) ** 2) / (2 * (8.0) ** 2))

    else: # pathological
        baseline = rng.choice([rng.uniform(102.0, 112.0), rng.uniform(172.0, 185.0)])
        stv = rng.uniform(0.6, 1.1)
        fhr = baseline + stv * np.sin(2 * np.pi * 0.02 * time_arr) + rng.normal(0, 0.9, num_points)
        dec_centers = [30.0, 75.0, 110.0]
        for center_t in dec_centers:
            if center_t < duration_sec:
                fhr -= 28.0 * np.exp(-((time_arr - center_t) ** 2) / (2 * (10.0) ** 2))

    fhr = np.clip(fhr, 60.0, 200.0)
    return time_arr, np.round(fhr, 2)


def setup_sample_data():
    sample_dir = Config.SAMPLE_DATA_DIR
    brain_dir = sample_dir / "brain_scans"
    cardiac_dir = sample_dir / "cardiac_signals"
    patients_dir = sample_dir / "patients"
    
    brain_dir.mkdir(parents=True, exist_ok=True)
    cardiac_dir.mkdir(parents=True, exist_ok=True)
    patients_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure Canonical Atlas Template exists
    get_default_atlas_template(Config.ATLAS_TEMPLATES_DIR, size=Config.IMAGE_TARGET_SIZE)
    print("[INIT] Canonical fetal brain atlas template cached.")

    # 2. Build 50 Synthetic Patient Cases
    patients_summary = []
    print("\n[GENERATING] Creating 50 synthetic patient profiles, matched brain scans & cardiac signals...")

    for i in range(50):
        pat_num = 1001 + i
        pat_id = f"PAT-{pat_num}"
        name = FICTIONAL_NAMES[i]
        
        if i < 30:
            category = "Normal"
            pattern = "normal"
            risk_score = round(float(0.06 + (i % 6) * 0.025), 3)
            risk_status = "Low Risk"
            expected_brain = "Normal Morphology"
            expected_cardiac = "Normal Reactive FHR"
        elif i < 42:
            category = "Suspect"
            pattern = "mild_variant" if (i % 2 == 0) else "suspect"
            risk_score = round(float(0.44 + (i % 5) * 0.045), 3)
            risk_status = "Requires Review"
            expected_brain = "Mild Ventricle Asymmetry / Variation"
            expected_cardiac = "Reduced Variability / Tachycardia"
        else:
            category = "Pathological"
            pattern = "pathological"
            risk_score = round(float(0.80 + (i % 4) * 0.042), 3)
            risk_status = "High Risk / Acute Intervention"
            expected_brain = "Pronounced Ventriculomegaly / Abnormality"
            expected_cardiac = "Late Decelerations / Severe Bradycardia"

        maternal_age = int(21 + (i * 7) % 21)
        gestational_week = round(float(24.0 + (i * 3.3) % 16.0), 1)
        est_weight_g = int(550 + (gestational_week - 24.0) * 180 + (i % 5) * 25)
        presentation = "Cephalic" if (i % 4 != 0) else ("Breech" if i % 4 == 1 else "Transverse")
        scan_type = "MRI" if (i % 2 == 0) else "Ultrasound"

        brain_filename = f"{pat_id}_brain.png"
        cardiac_filename = f"{pat_id}_fhr.csv"

        # 3. Generate Matched Brain Image
        brain_img_arr = generate_synthetic_brain_scan(
            gestational_week=gestational_week,
            pattern_type=pattern,
            scan_type=scan_type,
            seed=pat_num
        )
        brain_path = brain_dir / brain_filename
        Image.fromarray(brain_img_arr).save(str(brain_path))

        # 4. Generate Matched Cardiac Signal CSV
        time_arr, fhr_arr = generate_synthetic_cardiac_signal(
            gestational_week=gestational_week,
            pattern_type=pattern,
            duration_sec=120,
            sampling_rate=Config.CARDIAC_SAMPLING_RATE,
            seed=pat_num
        )
        baseline_fhr = round(float(np.median(fhr_arr)), 1)
        
        df_cardiac = pd.DataFrame({
            "time_seconds": np.round(time_arr, 2),
            "fetal_heart_rate": fhr_arr
        })
        cardiac_path = cardiac_dir / cardiac_filename
        df_cardiac.to_csv(str(cardiac_path), index=False)

        # 5. Create Detailed Patient Record JSON
        patient_dict = {
            "patient_id": pat_id,
            "name": name,
            "age": maternal_age,
            "maternal_age": maternal_age,
            "gestational_week": gestational_week,
            "gender": "Female",
            "fetal_heart_rate": baseline_fhr,
            "estimated_fetal_weight_g": est_weight_g,
            "presentation": presentation,
            "scan_type": scan_type,
            "brain_scan_filename": brain_filename,
            "cardiac_signal_filename": cardiac_filename,
            "expected_demo_result": category,
            "expected_brain_analysis": expected_brain,
            "expected_cardiac_analysis": expected_cardiac,
            "risk_score": risk_score,
            "risk_percent": round(risk_score * 100.0, 1),
            "risk_status": risk_status,
            "notes": f"Synthetic demo profile for {name}. Generated for academic AI multi-modal evaluation.",
            "data_tag": "SYNTHETIC DEMONSTRATION DATA — NOT FOR CLINICAL USE"
        }

        json_path = patients_dir / f"patient_{pat_num}.json"
        with open(json_path, "w") as jf:
            json.dump(patient_dict, jf, indent=2)

        patients_summary.append(patient_dict)

    # 6. Save Combined CSV Index (patients.csv)
    csv_index_path = sample_dir / "patients.csv"
    keys = list(patients_summary[0].keys())
    with open(csv_index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(patients_summary)

    # 7. Maintain Legacy Sample Files for backward compatibility
    legacy_brain = brain_dir / "sample_fetal_brain_mri.png"
    if not legacy_brain.exists():
        Image.fromarray(generate_synthetic_brain_scan(32.0, "normal", "MRI", seed=9999)).save(str(legacy_brain))
        
    legacy_norm = brain_dir / "fetal_brain_mri_30wk_normal.png"
    if not legacy_norm.exists():
        Image.fromarray(generate_synthetic_brain_scan(30.0, "normal", "MRI", seed=1001)).save(str(legacy_norm))

    legacy_var = brain_dir / "fetal_brain_mri_32wk_variant.png"
    if not legacy_var.exists():
        Image.fromarray(generate_synthetic_brain_scan(32.0, "mild_variant", "MRI", seed=1002)).save(str(legacy_var))

    legacy_us = brain_dir / "fetal_brain_us_28wk_scan.png"
    if not legacy_us.exists():
        Image.fromarray(generate_synthetic_brain_scan(28.0, "normal", "Ultrasound", seed=1003)).save(str(legacy_us))

    legacy_dat = cardiac_dir / "sample_ctu_chb_1001.dat"
    if not legacy_dat.exists():
        _, leg_fhr = generate_synthetic_cardiac_signal(32.0, "normal", 120, 4, seed=1001)
        np.savetxt(str(legacy_dat), leg_fhr, fmt="%.2f")

    # 8. Validation Summary
    print("\n" + "=" * 70)
    print("Successfully created 50 synthetic patient cases.")
    print("=" * 70)
    for p in patients_summary[:5]:
        print(f"{p['patient_id']} | {p['name']:<20} | {p['gestational_week']} weeks | Risk: {p['expected_demo_result']}")
    print("...")
    for p in patients_summary[-3:]:
        print(f"{p['patient_id']} | {p['name']:<20} | {p['gestational_week']} weeks | Risk: {p['expected_demo_result']}")
    print("-" * 70)
    print(f"Brain scans:       {len(list(brain_dir.glob('PAT-*_brain.png')))}")
    print(f"Cardiac signals:   {len(list(cardiac_dir.glob('PAT-*_fhr.csv')))}")
    print(f"Patient records:   {len(list(patients_dir.glob('patient_*.json')))}")
    print(f"Master CSV Index:  {csv_index_path}")
    print("All patient files successfully matched.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    setup_sample_data()
