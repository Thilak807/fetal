import torch
import torch.nn as nn
import torch.nn.functional as F

class FetalCardiacLSTM(nn.Module):
    """
    Bidirectional Long Short-Term Memory (LSTM) network with temporal attention
    for learning dynamic time-series representations from fetal cardiac signals.
    
    Exposes:
        extract_features(seq_tensor) -> Feature Vector F_cardiac in R^{feature_dim}
        forward(seq_tensor) -> class logits (when used standalone)
    """

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, feature_dim=128, num_classes=3, bidirectional=True):
        super(FetalCardiacLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Recurrent LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Temporal Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Temporal Feature Projection
        self.feature_projector = nn.Sequential(
            nn.Linear(lstm_output_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Standalone Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, x):
        """
        Extracts cardiac temporal feature vector F_cardiac prior to classification.
        Args:
            x: Tensor of shape (B, seq_len, input_dim)
        Returns:
            f_cardiac: Tensor of shape (B, feature_dim)
        """
        f_cardiac, _ = self.extract_features_with_attention(x)
        return f_cardiac

    def extract_features_with_attention(self, x):
        """
        Extracts cardiac feature vector along with temporal attention weights for explainability.
        Returns:
            f_cardiac: Tensor (B, feature_dim)
            att_weights: Tensor (B, seq_len)
        """
        out, (h_n, c_n) = self.lstm(x)
        att_logits = self.attention(out)
        att_weights = F.softmax(att_logits, dim=1)
        context = torch.sum(out * att_weights, dim=1)
        f_cardiac = self.feature_projector(context)
        return f_cardiac, att_weights.squeeze(-1)

    def forward(self, x):
        """Standard feedforward returning class logits."""
        f_cardiac = self.extract_features(x)
        logits = self.classifier(f_cardiac)
        return logits
