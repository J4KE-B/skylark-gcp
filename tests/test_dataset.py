import json
import cv2
import numpy as np
from src.dataset import GCPDataset
from src.transforms import get_val_transform

CLASS_TO_IDX = {"Cross": 0, "Square": 1, "L-Shape": 2}


def _make_fixture(tmp_path):
    root = tmp_path / "train"
    (root / "siteA" / "GCP1").mkdir(parents=True)
    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(root / "siteA" / "GCP1" / "a.JPG"), img)
    cv2.imwrite(str(root / "siteA" / "GCP1" / "b.JPG"), img)
    labels = {
        "siteA/GCP1/a.JPG": {"mark": {"x": 200.0, "y": 150.0}, "verified_shape": "Cross"},
        "siteA/GCP1/b.JPG": {"mark": {"x": 10.0, "y": 10.0}},  # missing shape -> impute Cross
    }
    lf = tmp_path / "labels.json"; lf.write_text(json.dumps(labels))
    return str(lf), str(root)


def test_len_imputes_missing_shape(tmp_path):
    lf, root = _make_fixture(tmp_path)
    ds = GCPDataset(lf, root, get_val_transform(768, 576), CLASS_TO_IDX, 768, 576, 4)
    assert len(ds) == 2  # both kept; second imputed from group majority


def test_item_contract(tmp_path):
    lf, root = _make_fixture(tmp_path)
    ds = GCPDataset(lf, root, get_val_transform(768, 576), CLASS_TO_IDX, 768, 576, 4)
    s = ds[0]
    assert s["image"].shape == (3, 576, 768)
    assert s["heatmap"].shape == (1, 144, 192)
    assert abs(s["heatmap"].sum().item() - 1.0) < 1e-4
    assert s["kp_norm"].shape == (2,) and -1 <= s["kp_norm"][0] <= 1
    assert s["cls_idx"].item() == 0
    assert tuple(s["orig_hw"].tolist()) == (300, 400)


def test_samples_expose_group(tmp_path):
    lf, root = _make_fixture(tmp_path)
    ds = GCPDataset(lf, root, get_val_transform(768, 576), CLASS_TO_IDX, 768, 576, 4)
    groups = {g for _, _, _, _, g in ds.samples}
    assert groups == {"siteA/GCP1"}
