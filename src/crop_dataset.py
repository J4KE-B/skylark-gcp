import json
from collections import Counter, defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def crop_size_for(h, w, frac=0.10, lo=256, hi=512):
    """Side length of the square crop, ~10% of the larger image dimension, clamped. Big enough
    to still contain the marker when localization is a bit off, small enough that the marker
    fills a useful fraction of the crop."""
    return int(min(hi, max(lo, frac * max(h, w))))


def crop_box(h, w, x, y, size):
    """Square box of side `size` centered at (x, y), shifted to stay inside the image (keeps the
    full size when the image is large enough; otherwise spans the whole dimension)."""
    half = size // 2
    if w >= size:
        x0 = max(0, min(int(round(x)) - half, w - size)); x1 = x0 + size
    else:
        x0, x1 = 0, w
    if h >= size:
        y0 = max(0, min(int(round(y)) - half, h - size)); y1 = y0 + size
    else:
        y0, y1 = 0, h
    return x0, y0, x1, y1


def crop_train_tf(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.8, border_mode=0, fill=0),   # shape class is rotation-invariant
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.7),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def crop_val_tf(size=224):
    return A.Compose([A.Resize(size, size), A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()])


def build_samples(label_file, root_dir, class_to_idx):
    """Same label parsing + folder-majority imputation as GCPDataset.
    Returns list of (rel, x, y, cls_idx, group)."""
    root = Path(root_dir)
    raw = json.load(open(label_file))
    folder_shapes = defaultdict(list)
    for k, v in raw.items():
        if "verified_shape" in v:
            folder_shapes[str(Path(k).parent)].append(v["verified_shape"])
    majority = {f: Counter(s).most_common(1)[0][0] for f, s in folder_shapes.items()}
    samples = []
    for rel, v in raw.items():
        if not (root / rel).exists():
            continue
        group = str(Path(rel).parent)
        shape = v.get("verified_shape") or majority.get(group)
        if shape not in class_to_idx:
            continue
        samples.append((rel, float(v["mark"]["x"]), float(v["mark"]["y"]),
                        class_to_idx[shape], group))
    return samples


class GCPCropDataset(Dataset):
    """Crops a square region around the GT marker from the ORIGINAL full-res image for shape
    classification. The whole-image model can't read the marker (it's sub-pixel at the encoder
    stride); a crop preserves the marker's detail. Training jitters the crop center so the
    classifier tolerates the localization error it sees at inference (predicted, not GT, center)."""

    def __init__(self, samples, root_dir, transform, crop_frac=0.10, jitter=0.15, train=True):
        self.samples = samples
        self.root = Path(root_dir)
        self.transform = transform
        self.crop_frac = crop_frac
        self.jitter = jitter
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, x, y, cls_idx, _ = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(self.root / rel)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        size = crop_size_for(h, w, self.crop_frac)
        if self.train and self.jitter > 0:
            j = self.jitter * size
            x = x + np.random.uniform(-j, j)
            y = y + np.random.uniform(-j, j)
        x0, y0, x1, y1 = crop_box(h, w, x, y, size)
        crop = img[y0:y1, x0:x1]
        out = self.transform(image=crop)
        return {"image": out["image"], "cls_idx": torch.tensor(cls_idx, dtype=torch.long)}
