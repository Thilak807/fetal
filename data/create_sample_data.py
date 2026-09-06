import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from scipy import ndimage

from backend.config import Config
from backend.atlas.deformable_registration import (
    generate_synthetic_fetal_brain_template,
    get_default_atlas_template,
)

def setup_sample_data():
    base_dir = Config.BASE_DIR
    sample_dir = Config.SAMPLE_DATA_DIR
    brain_dir = sample_dir / "brain_scans"
    cardiac_dir = sample_dir / "cardiac_signals"
    brain_dir.mkdir(parents=True, exist_ok=True)
    cardiac_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Canonical Atlas Template
    get_default_atlas_template(Config.ATLAS_TEMPLATES_DIR, size=Config.IMAGE_TARGET_SIZE)
    print("Canonical fetal brain atlas template cached.")

    # 2. Extract authentic CTU-CHB Cardiac Signals
    ctu_dat_dir = base_dir / "fetal-health-classification-main" / "CTU-CHB" / "dat"
    records = [1001, 1002, 1003, 1004]
    
    for rec in records:
        dat_file = ctu_dat_dir / f"{rec}.dat"
        hea_file = ctu_dat_dir / f"{rec}.hea"
        if dat_file.exists():
            d = np.fromfile(str(dat_file), dtype=np.int16).reshape(-1, 2)
            fhr = d[:, 0] / 100.0  # FHR in bpm
            
            # Save as CSV with time and FHR
            import pandas as pd
            time_sec = np.arange(len(fhr)) / 4.0
            df = pd.DataFrame({
                "time_seconds": np.round(time_sec, 2),
                "fetal_heart_rate": np.round(fhr, 2)
            })
            csv_path = cardiac_dir / f"ctu_chb_{rec}_fhr.csv"
            df.to_csv(str(csv_path), index=False)
            
            # Also save raw .dat
            dat_path = cardiac_dir / f"ctu_chb_{rec}.dat"
            np.savetxt(str(dat_path), fhr, fmt="%.2f")
            print(f"Extracted CTU-CHB record {rec}: {len(fhr)} points -> {csv_path.name}")

    # 3. Generate Representative Fetal Brain Scans (Normal & Variants)
    variations = [
        {"name": "fetal_brain_mri_30wk_normal.png", "angle": 0.0, "ventricle_scale": 1.0, "noise": 0.02},
        {"name": "fetal_brain_mri_32wk_variant.png", "angle": 4.0, "ventricle_scale": 1.15, "noise": 0.03},
        {"name": "fetal_brain_us_28wk_scan.png", "angle": -3.5, "ventricle_scale": 0.9, "noise": 0.04},
    ]

    for v in variations:
        scan = generate_synthetic_fetal_brain_template(size=Config.IMAGE_TARGET_SIZE)
        if v["angle"] != 0.0:
            scan = ndimage.rotate(scan, angle=v["angle"], reshape=False, mode='nearest')
        if v["noise"] > 0:
            scan = scan + np.random.normal(0, v["noise"], scan.shape)
        scan = np.clip(scan, 0.0, 1.0)
        uint8_img = (scan * 255.0).astype(np.uint8)
        img_path = brain_dir / v["name"]
        Image.fromarray(uint8_img).save(str(img_path))
        print(f"Generated sample brain scan: {v['name']}")

if __name__ == "__main__":
    setup_sample_data()
