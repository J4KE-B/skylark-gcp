import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CropClassifier(nn.Module):
    """Shape classifier over a high-res crop centered on the marker. Sees the marker at full
    detail — which the whole-image GCPModel cannot, since the ~35px marker is sub-pixel at the
    ResNet-50 encoder's stride-32 feature map. ResNet-18 is plenty for a 3-way shape decision."""

    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        in_f = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        self.net = base

    def forward(self, x):
        return self.net(x)
