import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.crop_dataset import crop_box, crop_size_for, crop_val_tf
from src.crop_model import CropClassifier
from src.dataset import GCPTestDataset
from src.model import GCPModel, softargmax2d
from src.transforms import get_val_transform
from src.metrics import coords_norm_to_orig_px


def format_prediction(x, y, shape):
    return {"mark": {"x": round(float(x), 2), "y": round(float(y), 2)}, "verified_shape": shape}


def reclassify_with_crops(preds, conf, cfg, crop_checkpoint, device, batch_size=32):
    """Stage 2: re-read each test image at full resolution, crop around the predicted (x, y),
    and classify the shape from that high-detail crop (flip-TTA averaged). Overwrites the
    coarse whole-image shape and records the shape confidence; the keypoint (x, y) is kept."""
    model = CropClassifier(len(cfg.classes), pretrained=False, dropout=cfg.model.dropout).to(device)
    ck = torch.load(crop_checkpoint, map_location=device)
    model.load_state_dict(ck["model_state"]); model.eval()
    tf = crop_val_tf()
    root = Path(cfg.paths.test_dir)
    keys = list(preds.keys())
    with torch.no_grad():
        for i in tqdm(range(0, len(keys), batch_size), desc="crop-cls"):
            chunk = keys[i:i + batch_size]
            tensors = []
            for k in chunk:
                img = cv2.cvtColor(cv2.imread(str(root / k)), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                size = crop_size_for(h, w)
                x0, y0, x1, y1 = crop_box(h, w, preds[k]["mark"]["x"], preds[k]["mark"]["y"], size)
                tensors.append(tf(image=img[y0:y1, x0:x1])["image"])
            xb = torch.stack(tensors).to(device)
            prob = (F.softmax(model(xb), 1)
                    + F.softmax(model(torch.flip(xb, [3])), 1)
                    + F.softmax(model(torch.flip(xb, [2])), 1)) / 3.0
            top = prob.max(1)
            for j, k in enumerate(chunk):
                preds[k]["verified_shape"] = cfg.idx_to_class[top.indices[j].item()]
                conf[k]["shape"] = round(top.values[j].item(), 4)
    return preds


def _coords_from_output(kp_out, kp_head):
    if kp_head == "heatmap":
        coords, _ = softargmax2d(kp_out)
        return coords
    return kp_out * 2.0 - 1.0


def peak_soft_refine(prob, window=5):
    """prob: (B,1,H,W) probability map (sums to 1 over HxW). Find the hard-argmax peak, then
    soft-argmax within a small window around it. This is robust to a second distractor blob
    (global soft-argmax averages toward the midpoint of two blobs -> catastrophic misses),
    while keeping sub-pixel precision. Returns coords (B,2) in [-1,1], matching softargmax2d."""
    b, _, h, w = prob.shape
    flat = prob.view(b, -1)
    idx = flat.argmax(dim=1)
    py = (idx // w).long()
    px = (idx % w).long()
    r = window // 2
    coords = torch.empty(b, 2, device=prob.device)
    for i in range(b):
        y0, y1 = int(max(0, py[i] - r)), int(min(h, py[i] + r + 1))
        x0, x1 = int(max(0, px[i] - r)), int(min(w, px[i] + r + 1))
        win = prob[i, 0, y0:y1, x0:x1]
        win = win / win.sum().clamp_min(1e-8)
        ys = torch.arange(y0, y1, device=prob.device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=prob.device, dtype=torch.float32)
        cy = (win.sum(1) * ys).sum()
        cx = (win.sum(0) * xs).sum()
        coords[i, 0] = cx / (w - 1) * 2.0 - 1.0
        coords[i, 1] = cy / (h - 1) * 2.0 - 1.0
    return coords


def peak_concentration(prob, window=7):
    """Localization confidence: fraction of heatmap probability mass within a window around the
    argmax peak. ~1.0 = one sharp marker (confident); low = diffuse/competing blobs (the model
    is unsure where the marker is — often a distractor grab or no clear marker). Returns (B,)."""
    b, _, h, w = prob.shape
    idx = prob.view(b, -1).argmax(dim=1)
    py = (idx // w).long(); px = (idx % w).long()
    r = window // 2
    out = torch.empty(b, device=prob.device)
    for i in range(b):
        y0, y1 = int(max(0, py[i] - r)), int(min(h, py[i] + r + 1))
        x0, x1 = int(max(0, px[i] - r)), int(min(w, px[i] + r + 1))
        out[i] = prob[i, 0, y0:y1, x0:x1].sum()
    return out


def predict(config_path, checkpoint_path, output_path="predictions.json", crop_checkpoint=None):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCPModel(len(cfg.classes), False, cfg.model.kp_head, cfg.model.dropout,
                     cfg.input.heatmap_stride).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    ds = GCPTestDataset(cfg.paths.test_dir, get_val_transform(cfg.input.width, cfg.input.height),
                        cfg.input.width, cfg.input.height)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers)
    preds = {}
    conf = {}
    with torch.no_grad():
        for b in tqdm(loader, desc="infer"):
            img = b["image"].to(device)
            views = [img]
            if cfg.predict.tta:
                views += [torch.flip(img, dims=[3]), torch.flip(img, dims=[2])]  # hflip, vflip
            heatmap_head = cfg.model.kp_head == "heatmap"
            kp_acc, cls_acc = [], []
            for vi, v in enumerate(views):
                kp_out, cls_logits = model(v)
                if heatmap_head:
                    # Accumulate in heatmap (probability) space, spatially un-flipped, so views
                    # that disagree don't get coordinate-averaged into the middle of nowhere.
                    bsz, _, hh, ww = kp_out.shape
                    prob = F.softmax(kp_out.view(bsz, -1), dim=1).view(bsz, 1, hh, ww)
                    if vi == 1:    # undo hflip
                        prob = torch.flip(prob, dims=[3])
                    elif vi == 2:  # undo vflip
                        prob = torch.flip(prob, dims=[2])
                    kp_acc.append(prob)
                else:
                    coords = kp_out * 2.0 - 1.0
                    if vi == 1:
                        coords = coords * torch.tensor([-1.0, 1.0], device=device)
                    elif vi == 2:
                        coords = coords * torch.tensor([1.0, -1.0], device=device)
                    kp_acc.append(coords)
                cls_acc.append(torch.softmax(cls_logits, dim=1))
            cls_prob = torch.stack(cls_acc).mean(0)
            cls_top = cls_prob.max(1)
            if heatmap_head:
                prob_mean = torch.stack(kp_acc).mean(0)        # average heatmaps, then localize once
                coords = peak_soft_refine(prob_mean).cpu()
                loc_conf = peak_concentration(prob_mean).cpu()
            else:
                coords = torch.stack(kp_acc).mean(0).cpu()
                loc_conf = torch.ones(img.size(0))             # regression head has no heatmap
            for i in range(img.size(0)):
                ox, oy = coords_norm_to_orig_px(coords[i:i + 1], cfg.input.width, cfg.input.height,
                                                b["scale"][i].item(),
                                                (b["pad"][i][0].item(), b["pad"][i][1].item()))
                path = b["path"][i]
                preds[path] = format_prediction(ox.item(), oy.item(),
                                                cfg.idx_to_class[cls_top.indices[i].item()])
                conf[path] = {"loc": round(loc_conf[i].item(), 4),
                              "shape": round(cls_top.values[i].item(), 4)}
    if crop_checkpoint:
        print("Stage 2: reclassifying shapes from high-res crops...")
        preds = reclassify_with_crops(preds, conf, cfg, crop_checkpoint, device)
    json.dump(preds, open(output_path, "w"), indent=2)
    print(f"wrote {len(preds)} predictions -> {output_path}")
    # Confidence is written to a SEPARATE file so predictions.json keeps the exact label schema.
    conf_path = str(Path(output_path).with_name("confidences.json"))
    json.dump(conf, open(conf_path, "w"), indent=2)
    print(f"wrote confidences -> {conf_path} (loc = heatmap sharpness, shape = classifier prob)")
    return preds
