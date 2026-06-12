import torch
from src.losses import GCPLoss


def test_gcploss_heatmap_decreases_on_perfect_match():
    loss = GCPLoss(kp_weight=1.0, cls_weight=0.5, reg_weight=1.0,
                   class_weights=None, kp_head="heatmap")
    # build a batch whose target equals a sharp center peak
    hm = torch.full((2, 1, 144, 192), -10.0); hm[:, 0, 72, 96] = 10.0
    target_hm = torch.zeros(2, 1, 144, 192); target_hm[:, 0, 72, 96] = 1.0
    batch = {
        "kp_norm": torch.zeros(2, 2),
        "heatmap": target_hm,
        "cls_idx": torch.tensor([0, 1]),
    }
    cls_logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0]])
    total, kp_l, cls_l, coords = loss(hm, cls_logits, batch)
    assert total.item() >= 0
    assert coords.shape == (2, 2)
    assert kp_l.item() < 0.1  # near-perfect keypoint match
