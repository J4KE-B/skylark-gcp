import numpy as np
from src.transforms import get_train_transform, get_val_transform


def test_train_transform_outputs_tensor_and_keypoint():
    tf = get_train_transform(768, 576)
    img = np.full((576, 768, 3), 128, dtype=np.uint8)
    out = tf(image=img, keypoints=[(400.0, 300.0)])
    assert out["image"].shape == (3, 576, 768)
    assert len(out["keypoints"]) == 1


def test_val_transform_is_clean_resize_only():
    tf = get_val_transform(768, 576)
    img = np.full((576, 768, 3), 128, dtype=np.uint8)
    out = tf(image=img, keypoints=[(400.0, 300.0)])
    assert out["image"].shape == (3, 576, 768)
    assert abs(out["keypoints"][0][0] - 400.0) < 1e-3  # val does not move the point
