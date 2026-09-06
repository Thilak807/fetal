import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_model import FetalBrainCNN
from .lstm_model import FetalCardiacLSTM

class MultiModalFusionModel(nn.Module):
    """
    Multi-Modal Fetal Risk Assessment Neural Network.
    
    Combines:
        1. Fetal Brain Spatial Features (F_brain from FetalBrainCNN)
        2. Fetal Cardiac Temporal Features (F_cardiac from FetalCardiacLSTM)
    
    Fusion Mechanism:
        F_concat = [F_brain || F_cardiac]
        Gated Modality Attention: g = sigmoid(Linear(F_concat))
        F_fused = F_concat * g
    
    Risk Assessment:
        F_fused -> MLP Classifier -> Class Logits & Probabilities
        Risk Score = Continuous clinical risk index in [0.0, 1.0]
    """

    CLASS_LABELS_3 = ["Normal", "Suspect", "Pathological"]
    CLASS_LABELS_2 = ["Normal", "At-Risk (Hypoxia)"]

    def __init__(
        self,
        brain_feature_dim=128,
        cardiac_feature_dim=128,
        fusion_hidden_dim=128,
        num_classes=3,
        use_gated_attention=True,
    ):
        super(MultiModalFusionModel, self).__init__()
        self.brain_feature_dim = brain_feature_dim
        self.cardiac_feature_dim = cardiac_feature_dim
        self.total_feature_dim = brain_feature_dim + cardiac_feature_dim  # e.g., 256
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_classes = num_classes
        self.use_gated_attention = use_gated_attention

        # Feature Extractors
        self.brain_cnn = FetalBrainCNN(in_channels=1, feature_dim=brain_feature_dim, num_classes=num_classes)
        self.cardiac_lstm = FetalCardiacLSTM(input_dim=1, feature_dim=cardiac_feature_dim, num_classes=num_classes)

        # Modality Gating Network
        if self.use_gated_attention:
            self.gate = nn.Sequential(
                nn.Linear(self.total_feature_dim, self.total_feature_dim),
                nn.Sigmoid()
            )

        # Multi-Modal Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.total_feature_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def fuse_features(self, f_brain, f_cardiac):
        """
        Concatenates and applies gated attention over the two modality vectors.
        Args:
            f_brain: Tensor (B, brain_feature_dim)
            f_cardiac: Tensor (B, cardiac_feature_dim)
        Returns:
            f_fused: Tensor (B, total_feature_dim)
            gates: Tensor (B, total_feature_dim)
        """
        # Feature-level concatenation
        f_concat = torch.cat([f_brain, f_cardiac], dim=-1)

        if self.use_gated_attention:
            gates = self.gate(f_concat)
            f_fused = f_concat * gates
        else:
            gates = torch.ones_like(f_concat)
            f_fused = f_concat

        return f_fused, gates

    def forward(self, brain_images, cardiac_sequences):
        """
        End-to-end forward pass:
        Inputs:
            brain_images: Tensor (B, 1, 128, 128)
            cardiac_sequences: Tensor (B, seq_len, 1)
        Outputs:
            logits: Tensor (B, num_classes)
            f_brain: Tensor (B, brain_feature_dim)
            f_cardiac: Tensor (B, cardiac_feature_dim)
            gates: Tensor (B, total_feature_dim)
        """
        f_brain = self.brain_cnn.extract_features(brain_images)
        f_cardiac = self.cardiac_lstm.extract_features(cardiac_sequences)

        f_fused, gates = self.fuse_features(f_brain, f_cardiac)
        logits = self.fusion_mlp(f_fused)

        return logits, f_brain, f_cardiac, gates

    def compute_risk_score(self, probabilities):
        """
        Computes continuous risk assessment index (0.0 to 1.0)
        from class probability distribution.
        """
        if self.num_classes == 3:
            # Normal: 0.0, Suspect: 0.5, Pathological: 1.0
            weights = torch.tensor([0.0, 0.5, 1.0], device=probabilities.device)
            risk_score = torch.sum(probabilities * weights, dim=-1)
        else:
            # Binary: [Normal, At-Risk]
            risk_score = probabilities[:, 1]
        return risk_score
