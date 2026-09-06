import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from backend.preprocessing.cardiac_preprocessing import CardiacPreprocessor
from backend.preprocessing.brain_preprocessing import BrainPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.cardiac_preprocessor = CardiacPreprocessor(sampling_rate=4, window_len=120)
        self.brain_preprocessor = BrainPreprocessor(target_size=(128, 128))

    def test_cardiac_extreme_values(self):
        """Tests that non-physiological FHR values (<50 or >200 bpm) are interpolated."""
        raw_sig = np.array([135.0, 138.0, 240.0, 137.0, 30.0, 136.0, 135.0])
        clean = self.cardiac_preprocessor.remove_extreme_values(raw_sig)
        self.assertTrue(np.all(clean >= 50.0))
        self.assertTrue(np.all(clean <= 200.0))

    def test_cardiac_missing_values(self):
        """Tests that zero dropouts are correctly interpolated."""
        raw_sig = np.array([130.0, 132.0, 0.0, 0.0, 135.0, 136.0])
        clean = self.cardiac_preprocessor.replace_missing_values(raw_sig)
        self.assertTrue(np.all(clean > 0.0))
        self.assertAlmostEqual(clean[2], 133.0, places=1)

    def test_cardiac_sequence_preparation(self):
        """Tests LSTM sequence window formatting (N, window_len, 1)."""
        sig = np.linspace(120, 150, 300)
        seqs = self.cardiac_preprocessor.prepare_sequences(sig, window_len=120)
        self.assertEqual(seqs.ndim, 3)
        self.assertEqual(seqs.shape[1], 120)
        self.assertEqual(seqs.shape[2], 1)

    def test_brain_normalization_and_resize(self):
        """Tests brain image intensity normalization and canonical 128x128 grid."""
        synthetic_raw = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        norm, metrics = self.brain_preprocessor.preprocess(synthetic_raw)
        self.assertEqual(norm.shape, (128, 128))
        self.assertGreaterEqual(float(np.min(norm)), 0.0)
        self.assertLessEqual(float(np.max(norm)), 1.0)
        self.assertIn("target_resolution", metrics)

if __name__ == "__main__":
    unittest.main()
