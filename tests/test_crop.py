import torch

from src.crop_dataset import crop_box, crop_size_for
from src.crop_model import CropClassifier


def test_crop_classifier_shape():
    m = CropClassifier(num_classes=3, pretrained=False)
    out = m(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 3)


def test_crop_box_full_size_and_in_bounds_near_corner():
    h, w = 3000, 4000
    size = crop_size_for(h, w)
    x0, y0, x1, y1 = crop_box(h, w, 5, 5, size)   # marker near top-left corner
    assert x1 - x0 == size and y1 - y0 == size    # full square preserved
    assert 0 <= x0 and x1 <= w and 0 <= y0 and y1 <= h


def test_crop_box_centered_when_room():
    h, w = 3000, 4000
    size = crop_size_for(h, w)
    cx, cy = 2000, 1500
    x0, y0, x1, y1 = crop_box(h, w, cx, cy, size)
    assert abs((x0 + x1) // 2 - cx) <= 1 and abs((y0 + y1) // 2 - cy) <= 1


def test_crop_size_clamped():
    assert crop_size_for(4000, 4000) <= 512
    assert crop_size_for(400, 400) >= 256
