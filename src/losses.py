import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import softargmax2d


class GCPLoss(nn.Module):
    def __init__(self, kp_weight=1.0, cls_weight=0.5, reg_weight=1.0,
                 class_weights=None, kp_head="heatmap"):
        super().__init__()
        self.kp_weight = kp_weight
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.kp_head = kp_head
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None else None)

    def forward(self, kp_out, cls_logits, batch):
        if self.kp_head == "heatmap":
            coords, prob = softargmax2d(kp_out)
            coord_loss = F.mse_loss(coords, batch["kp_norm"])
            reg = F.mse_loss(prob, batch["heatmap"])
            kp_loss = coord_loss + self.reg_weight * reg
        else:
            kp_loss = F.l1_loss(kp_out, (batch["kp_norm"] + 1.0) / 2.0)
            coords = kp_out * 2.0 - 1.0
        cls_loss = F.cross_entropy(cls_logits, batch["cls_idx"], weight=self.class_weights)
        total = self.kp_weight * kp_loss + self.cls_weight * cls_loss
        return total, kp_loss, cls_loss, coords
