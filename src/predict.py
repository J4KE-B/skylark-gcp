import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import GCPTestDataset
from src.model import GCPModel, softargmax2d
from src.transforms import get_val_transform
from src.metrics import coords_norm_to_orig_px


def format_prediction(x, y, shape):
    return {"mark": {"x": round(float(x), 2), "y": round(float(y), 2)}, "verified_shape": shape}


def _coords_from_output(kp_out, kp_head):
    if kp_head == "heatmap":
        coords, _ = softargmax2d(kp_out)
        return coords
    return kp_out * 2.0 - 1.0


def predict(config_path, checkpoint_path, output_path="predictions.json"):
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
    with torch.no_grad():
        for b in tqdm(loader, desc="infer"):
            img = b["image"].to(device)
            views = [img]
            if cfg.predict.tta:
                views += [torch.flip(img, dims=[3]), torch.flip(img, dims=[2])]  # hflip, vflip
            coords_acc, cls_acc = [], []
            for vi, v in enumerate(views):
                kp_out, cls_logits = model(v)
                coords = _coords_from_output(kp_out, cfg.model.kp_head)  # (B,2) in [-1,1]
                if vi == 1:   # undo hflip: x -> -x
                    coords = coords * torch.tensor([-1.0, 1.0], device=device)
                elif vi == 2:  # undo vflip: y -> -y
                    coords = coords * torch.tensor([1.0, -1.0], device=device)
                coords_acc.append(coords); cls_acc.append(torch.softmax(cls_logits, dim=1))
            coords = torch.stack(coords_acc).mean(0).cpu()
            cls_idx = torch.stack(cls_acc).mean(0).argmax(1).cpu()
            for i in range(img.size(0)):
                ox, oy = coords_norm_to_orig_px(coords[i:i + 1], cfg.input.width, cfg.input.height,
                                                b["scale"][i].item(),
                                                (b["pad"][i][0].item(), b["pad"][i][1].item()))
                preds[b["path"][i]] = format_prediction(ox.item(), oy.item(),
                                                        cfg.idx_to_class[cls_idx[i].item()])
    json.dump(preds, open(output_path, "w"), indent=2)
    print(f"wrote {len(preds)} predictions -> {output_path}")
    return preds
