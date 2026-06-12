import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


def softargmax2d(heatmap_logits: torch.Tensor):
    """heatmap_logits: (B,1,H,W). Returns (coords (B,2) in [-1,1], prob (B,1,H,W) summing to 1)."""
    b, _, h, w = heatmap_logits.shape
    prob = F.softmax(heatmap_logits.view(b, -1), dim=1).view(b, 1, h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=heatmap_logits.device)
    ys = torch.linspace(-1.0, 1.0, h, device=heatmap_logits.device)
    px = (prob.sum(dim=2).squeeze(1) * xs).sum(dim=1)  # (B,)
    py = (prob.sum(dim=3).squeeze(1) * ys).sum(dim=1)  # (B,)
    return torch.stack([px, py], dim=1), prob


class GCPModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, kp_head="heatmap",
                 dropout=0.3, heatmap_stride=4):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-2])  # (B,2048,H/32,W/32)
        self.kp_head = kp_head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(2048, num_classes),
        )
        if kp_head == "heatmap":
            up = 32 // heatmap_stride  # /32 feature -> /stride heatmap
            self.kp_decoder = nn.Sequential(
                nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=up, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 1, 1),
            )
        else:
            self.kp_reg = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(dropout), nn.Linear(2048, 2), nn.Sigmoid(),
            )

    def forward(self, x):
        feat = self.encoder(x)
        cls = self.cls_head(feat)
        if self.kp_head == "heatmap":
            return self.kp_decoder(feat), cls
        return self.kp_reg(feat), cls
