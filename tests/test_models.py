import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from backend.models.cnn_model import FetalBrainCNN
from backend.models.lstm_model import FetalCardiacLSTM
from backend.models.fusion_model import MultiModalFusionModel

class TestNeuralModels(unittest.TestCase):
    def test_brain_cnn(self):
        """Tests Brain CNN feature extraction and forward pass."""
        model = FetalBrainCNN(in_channels=1, feature_dim=128, num_classes=3)
        dummy_scan = torch.randn(2, 1, 128, 128)
        
        # Test feature extraction
        f_brain = model.extract_features(dummy_scan)
        self.assertEqual(f_brain.shape, (2, 128))
        self.assertFalse(torch.isnan(f_brain).any())

        # Test forward logits
        logits = model(dummy_scan)
        self.assertEqual(logits.shape, (2, 3))

    def test_cardiac_lstm(self):
        """Tests Cardiac LSTM feature extraction and temporal attention."""
        model = FetalCardiacLSTM(input_dim=1, hidden_dim=64, feature_dim=128, num_classes=3)
        dummy_seq = torch.randn(2, 120, 1)
        
        # Test feature extraction
        f_cardiac = model.extract_features(dummy_seq)
        self.assertEqual(f_cardiac.shape, (2, 128))
        self.assertFalse(torch.isnan(f_cardiac).any())

        # Test forward logits
        logits = model(dummy_seq)
        self.assertEqual(logits.shape, (2, 3))

    def test_multimodal_fusion(self):
        """Tests feature-level fusion, gated attention, and continuous risk scoring."""
        model = MultiModalFusionModel(
            brain_feature_dim=128,
            cardiac_feature_dim=128,
            fusion_hidden_dim=128,
            num_classes=3,
            use_gated_attention=True
        )
        dummy_scan = torch.randn(2, 1, 128, 128)
        dummy_seq = torch.randn(2, 120, 1)

        logits, f_brain, f_cardiac, gates = model(dummy_scan, dummy_seq)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(f_brain.shape, (2, 128))
        self.assertEqual(f_cardiac.shape, (2, 128))
        self.assertEqual(gates.shape, (2, 256))

        # Test continuous risk scoring
        probs = torch.softmax(logits, dim=-1)
        risk_score = model.compute_risk_score(probs)
        self.assertEqual(risk_score.shape, (2,))
        self.assertTrue(torch.all(risk_score >= 0.0) and torch.all(risk_score <= 1.0))

if __name__ == "__main__":
    unittest.main()
