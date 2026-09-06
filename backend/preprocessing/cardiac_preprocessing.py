import os
import numpy as np
import pandas as pd
from scipy import interpolate

class CardiacPreprocessor:
    """
    Fetal cardiac signal preprocessor adapted from the CTU-CHB preprocessing
    pipeline (CTU-CHB/simple_denoise.py & Feature_Extraction.py).
    
    Provides:
    1. Robust file loading (CSV, DAT, TXT)
    2. Missing value detection & linear interpolation
    3. Non-physiological extreme value filtering (50 - 200 bpm)
    4. Sudden artifact / large jump smoothing (>25 bpm delta)
    5. Baseline FHR, Short-Term Variability (STV), and Long-Term Variability (LTV) extraction
    6. Sequence/window generation for LSTM temporal feature modeling
    7. JSON-ready waveform subsampling for web UI visualization
    """

    def __init__(self, sampling_rate=4, min_hr=50.0, max_hr=200.0, max_delta=25.0, window_len=120):
        self.sampling_rate = sampling_rate  # 4 Hz default
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.max_delta = max_delta
        self.window_len = window_len  # 120 points = 30s window

    def load_signal(self, file_path):
        """Loads FHR signal from CSV, DAT, or TXT file."""
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Signal file not found: {file_path}")

        ext = file_path.split('.')[-1].lower()

        if ext == 'csv':
            try:
                df = pd.read_csv(file_path)
                # Look for common CTG column names
                candidate_cols = [
                    'fhr', 'fetal_heart_rate', 'heart_rate', 'signal', 
                    'fetal_hr', 'baseline value', 'bpm'
                ]
                selected_col = None
                for col in df.columns:
                    if col.lower().strip() in candidate_cols:
                        selected_col = col
                        break
                if selected_col is not None:
                    raw_sig = df[selected_col].dropna().values.astype(np.float64)
                else:
                    # Default to the first numeric column
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        raw_sig = df[num_cols[0]].dropna().values.astype(np.float64)
                    else:
                        raise ValueError("No numeric signal column found in CSV.")
            except Exception as e:
                raise ValueError(f"Failed to parse CSV signal: {e}")

        elif ext in ['dat', 'txt']:
            try:
                raw_sig = np.loadtxt(file_path, dtype=np.float64)
                if raw_sig.ndim > 1:
                    raw_sig = raw_sig[:, 0]
            except Exception:
                # Try reading line-by-line in case of whitespace/header anomalies
                values = []
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        try:
                            values.append(float(parts[0]))
                        except ValueError:
                            continue
                if not values:
                    raise ValueError("Could not parse numeric cardiac values from file.")
                raw_sig = np.array(values, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported cardiac file extension: {ext}")

        if len(raw_sig) < 10:
            raise ValueError("Signal too short: minimum 10 sample points required.")

        return raw_sig

    def remove_extreme_values(self, sig):
        """Removes values outside physiologically plausible fetal HR bounds [50, 200] bpm."""
        clean_sig = np.copy(sig)
        mask = (clean_sig >= self.min_hr) & (clean_sig <= self.max_hr)
        n_valid = np.sum(mask)

        if 0 < n_valid < len(mask):
            x = np.arange(len(mask))
            clean_sig = np.interp(x, x[mask], clean_sig[mask])
        elif n_valid == 0:
            # Fallback if entire window was zero or noisy
            clean_sig.fill(130.0)  # Standard normal baseline fallback

        return clean_sig

    def replace_missing_values(self, sig):
        """Detects zeros or NaNs and applies linear interpolation."""
        clean_sig = np.copy(sig)
        clean_sig = np.nan_to_num(clean_sig, nan=0.0)
        valid = clean_sig > 0
        n_valid = np.sum(valid)

        if 0 < n_valid < len(clean_sig):
            x = np.arange(len(clean_sig))
            clean_sig = np.interp(x, x[valid], clean_sig[valid])
        elif n_valid == 0:
            clean_sig.fill(130.0)

        return clean_sig

    def filter_large_changes(self, sig):
        """Smooths unphysiological jumps (> max_delta bpm per sample step)."""
        clean_sig = np.copy(sig)
        diff = np.abs(np.diff(clean_sig))
        change_mask = diff > self.max_delta

        if np.any(change_mask):
            # Pad mask to match signal length
            pad_mask = np.zeros(len(clean_sig), dtype=bool)
            pad_mask[1:] = change_mask
            valid_idx = np.where(~pad_mask)[0]
            if len(valid_idx) > 1:
                x = np.arange(len(clean_sig))
                clean_sig = np.interp(x, x[valid_idx], clean_sig[valid_idx])

        return clean_sig

    def calculate_statistics(self, sig):
        """Calculates clinical FHR statistics: baseline, STV, LTV, accelerations, decelerations."""
        baseline = float(np.mean(sig))
        std_val = float(np.std(sig))
        
        # Short-term variability (mean absolute difference between successive beats/samples)
        diffs = np.abs(np.diff(sig))
        stv = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        
        # Long-term variability (range over 1-minute windows)
        pts_per_min = self.sampling_rate * 60
        if len(sig) >= pts_per_min:
            n_min_windows = len(sig) // pts_per_min
            ranges = [
                np.ptp(sig[i * pts_per_min : (i + 1) * pts_per_min])
                for i in range(n_min_windows)
            ]
            ltv = float(np.mean(ranges))
        else:
            ltv = float(np.ptp(sig))

        # Detect accelerations (>15 bpm above baseline for >15s) and decelerations
        acc_threshold = baseline + 15.0
        dec_threshold = baseline - 15.0
        n_acc = int(np.sum(sig > acc_threshold) // (self.sampling_rate * 15))
        n_dec = int(np.sum(sig < dec_threshold) // (self.sampling_rate * 15))

        return {
            "baseline_fhr": round(baseline, 2),
            "std_fhr": round(std_val, 2),
            "short_term_variability": round(stv, 3),
            "long_term_variability": round(ltv, 3),
            "min_fhr": round(float(np.min(sig)), 2),
            "max_fhr": round(float(np.max(sig)), 2),
            "accelerations_count": n_acc,
            "decelerations_count": n_dec,
            "sample_count": len(sig),
            "duration_seconds": round(len(sig) / self.sampling_rate, 2),
        }

    def preprocess(self, raw_signal):
        """
        Executes complete denoising and conditioning pipeline.
        Returns:
            processed_signal: np.ndarray
            stats: dict
        """
        sig = self.replace_missing_values(raw_signal)
        sig = self.remove_extreme_values(sig)
        sig = self.filter_large_changes(sig)
        stats = self.calculate_statistics(sig)
        return sig, stats

    def prepare_sequences(self, sig, window_len=None, step=None):
        """
        Prepares fixed-length normalized sequences for LSTM input tensor [N, seq_len, 1].
        Standardizes input around baseline (130 bpm) and scale (25 bpm).
        """
        if window_len is None:
            window_len = self.window_len
        if step is None:
            step = window_len // 2  # 50% overlap

        # Normalize signal: (FHR - 130) / 25
        norm_sig = (sig - 130.0) / 25.0

        if len(norm_sig) < window_len:
            # Pad with repeating edge values
            padded = np.pad(norm_sig, (0, window_len - len(norm_sig)), mode='edge')
            sequences = np.expand_dims(padded, axis=0)  # shape (1, window_len)
        else:
            seqs = []
            for i in range(0, len(norm_sig) - window_len + 1, max(1, step)):
                seqs.append(norm_sig[i : i + window_len])
            if not seqs:
                seqs.append(norm_sig[:window_len])
            sequences = np.array(seqs)

        # Add feature dimension: shape (N, window_len, 1)
        sequences = np.expand_dims(sequences, axis=-1).astype(np.float32)
        return sequences

    def get_visualization_payload(self, raw_signal, processed_signal, max_points=300):
        """
        Generates subsampled time-series points suitable for lightweight web charts.
        """
        n_points = len(processed_signal)
        stride = max(1, n_points // max_points)
        indices = np.arange(0, n_points, stride)

        time_axis = [round(float(i / self.sampling_rate), 2) for i in indices]
        raw_vals = [round(float(raw_signal[i]), 1) for i in indices]
        proc_vals = [round(float(processed_signal[i]), 1) for i in indices]

        return {
            "time": time_axis,
            "raw": raw_vals,
            "processed": proc_vals,
            "baseline": float(np.mean(processed_signal)),
            "unit": "bpm",
            "time_unit": "seconds",
        }
