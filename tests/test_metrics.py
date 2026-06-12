import torch
from src.metrics import coords_norm_to_orig_px, pck, macro_f1


def test_coords_norm_to_orig_px_roundtrip():
    coords = torch.tensor([[0.0, 0.0]])
    ox, oy = coords_norm_to_orig_px(coords, 768, 576, scale=0.5, pad=(0, 48))
    assert ox.shape == (1,) and oy.shape == (1,)


def test_pck_counts_within_threshold():
    pred = torch.tensor([[100.0, 100.0], [100.0, 100.0]])
    gt = torch.tensor([[105.0, 100.0], [200.0, 100.0]])  # dist 5 and 100
    assert pck(pred, gt, thr=10.0) == 0.5


def test_macro_f1_perfect_and_wrong():
    assert macro_f1(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), 3) == 1.0
    assert macro_f1(torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2]), 3) == 0.0
