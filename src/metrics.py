import torch
from sklearn.metrics import f1_score


def coords_norm_to_orig_px(coords_norm, input_w, input_h, scale, pad):
    """(B,2) normalized [-1,1] over letterbox input -> original-pixel (ox, oy) tensors (B,)."""
    px = (coords_norm[:, 0] + 1.0) / 2.0 * (input_w - 1)
    py = (coords_norm[:, 1] + 1.0) / 2.0 * (input_h - 1)
    ox = (px - pad[0]) / scale
    oy = (py - pad[1]) / scale
    return ox, oy


def pck(pred_px_xy, gt_px_xy, thr):
    d = torch.norm(pred_px_xy - gt_px_xy, dim=1)
    return (d < thr).float().mean().item()


def macro_f1(pred_idx, gt_idx, num_classes):
    return f1_score(gt_idx.cpu().numpy(), pred_idx.cpu().numpy(),
                    labels=list(range(num_classes)), average="macro", zero_division=0)
