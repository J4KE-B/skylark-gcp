import torch
from src.model import GCPModel, softargmax2d


def test_heatmap_forward_shapes():
    m = GCPModel(num_classes=3, pretrained=False, kp_head="heatmap", heatmap_stride=4)
    x = torch.rand(2, 3, 576, 768)
    hm, cls = m(x)
    assert hm.shape == (2, 1, 144, 192)
    assert cls.shape == (2, 3)


def test_softargmax_center_and_prob_sums_to_one():
    # a sharp peak at center should decode to ~ (0,0) in [-1,1]
    hm = torch.full((1, 1, 144, 192), -10.0)
    hm[0, 0, 72, 96] = 10.0
    coords, prob = softargmax2d(hm)
    assert torch.allclose(prob.sum(dim=(1, 2, 3)), torch.ones(1), atol=1e-4)
    assert abs(coords[0, 0].item()) < 0.05 and abs(coords[0, 1].item()) < 0.05


def test_regression_head_in_unit_range():
    m = GCPModel(num_classes=3, pretrained=False, kp_head="regression")
    xy, cls = m(torch.rand(2, 3, 576, 768))
    assert xy.shape == (2, 2) and xy.min() >= 0 and xy.max() <= 1
