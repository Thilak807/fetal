import torch
import torch.nn as nn
import torch.nn.functional as F

class FetalBrainCNN(nn.Module):
    """
    Deep Convolutional Neural Network for spatial feature extraction
    from atlas-mapped fetal brain imaging.
    
    Exposes:
        extract_features(image_tensor) -> Feature Vector F_brain in R^{feature_dim}
        forward(image_tensor) -> class logits (when used standalone)
    """

    def __init__(self, in_channels=1, feature_dim=128, num_classes=3):
        super(FetalBrainCNN, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Convolutional Backbone (4 stages)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Spatial Feature Projection
        self.feature_projector = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Standalone Classification Head (used during isolated brain training)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, x):
        """
        Extracts brain spatial feature vector F_brain prior to classification.
        Args:
            x: Tensor of shape (B, in_channels, H, W)
        Returns:
            f_brain: Tensor of shape (B, feature_dim)
        """
        # Block 1: 128x128 -> 64x64
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2: 64x64 -> 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Block 4: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Global average pooling: (B, 256, 1, 1) -> (B, 256)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Spatial feature vector F_brain: (B, feature_dim)
        f_brain = self.feature_projector(x)
        return f_brain

    def forward(self, x):
        """Standard feedforward returning class logits."""
        f_brain = self.extract_features(x)
        logits = self.classifier(f_brain)
        return logits
