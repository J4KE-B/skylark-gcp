import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.letterbox import letterbox, fwd_point


def _gaussian_heatmap(h, w, cx, cy, sigma=2.0):
    ys, xs = np.mgrid[0:h, 0:w]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    s = g.sum()
    return (g / s).astype(np.float32) if s > 0 else g.astype(np.float32)


class GCPDataset(Dataset):
    def __init__(self, label_file, root_dir, transform, class_to_idx,
                 input_w=768, input_h=576, heatmap_stride=4, sigma=2.0):
        self.root = Path(root_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.input_w, self.input_h, self.stride, self.sigma = input_w, input_h, heatmap_stride, sigma

        raw = json.load(open(label_file))
        # folder-majority shape for imputation
        folder_shapes = defaultdict(list)
        for k, v in raw.items():
            if "verified_shape" in v:
                folder_shapes[str(Path(k).parent)].append(v["verified_shape"])
        majority = {f: Counter(s).most_common(1)[0][0] for f, s in folder_shapes.items()}

        self.samples = []
        for rel, v in raw.items():
            img_path = self.root / rel
            if not img_path.exists():
                continue
            group = str(Path(rel).parent)
            shape = v.get("verified_shape") or majority.get(group)
            if shape not in class_to_idx:
                continue  # unlabelled and no majority -> drop
            self.samples.append((rel, float(v["mark"]["x"]), float(v["mark"]["y"]),
                                 class_to_idx[shape], group))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, x, y, cls_idx, _ = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(self.root / rel)), cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]
        lb, scale, pad = letterbox(img, self.input_w, self.input_h)
        kx, ky = fwd_point((x, y), scale, pad)
        out = self.transform(image=lb, keypoints=[(kx, ky)])
        image = out["image"]
        if len(out["keypoints"]) > 0:
            kx, ky = out["keypoints"][0]
        kx = float(np.clip(kx, 0, self.input_w - 1)); ky = float(np.clip(ky, 0, self.input_h - 1))

        hw, hh = self.input_w // self.stride, self.input_h // self.stride
        heatmap = _gaussian_heatmap(hh, hw, kx / self.stride, ky / self.stride, self.sigma)
        kp_norm = torch.tensor([kx / (self.input_w - 1) * 2 - 1,
                                ky / (self.input_h - 1) * 2 - 1], dtype=torch.float32)
        return {
            "image": image,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "kp_norm": kp_norm,
            "cls_idx": torch.tensor(cls_idx, dtype=torch.long),
            "orig_hw": torch.tensor([oh, ow], dtype=torch.long),
            "scale": float(scale),
            "pad": torch.tensor([pad[0], pad[1]], dtype=torch.long),
            "path": rel,
        }


class GCPTestDataset(Dataset):
    def __init__(self, test_dir, transform, input_w=768, input_h=576):
        self.root = Path(test_dir)
        self.transform = transform
        self.input_w, self.input_h = input_w, input_h
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]
        lb, scale, pad = letterbox(img, self.input_w, self.input_h)
        out = self.transform(image=lb, keypoints=[(0.0, 0.0)])
        return {
            "image": out["image"],
            "orig_hw": torch.tensor([oh, ow], dtype=torch.long),
            "scale": float(scale),
            "pad": torch.tensor([pad[0], pad[1]], dtype=torch.long),
            "path": str(p.relative_to(self.root)),
        }
