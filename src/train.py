import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import load_config
from src.dataset import GCPDataset
from src.model import GCPModel
from src.transforms import get_train_transform, get_val_transform
from src.losses import GCPLoss
from src.metrics import coords_norm_to_orig_px, pck, macro_f1
from src.splits import group_stratified_split


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def class_weights_from(samples, num_classes):
    counts = Counter(c for _, _, _, c, _ in samples)
    total = sum(counts.values())
    return torch.tensor([total / (num_classes * max(1, counts.get(i, 0)))
                         for i in range(num_classes)], dtype=torch.float32)


def _gt_orig_px(b, cfg):
    out = []
    for i in range(b["kp_norm"].size(0)):
        nx, ny = b["kp_norm"][i].tolist()
        px = (nx + 1) / 2 * (cfg.input.width - 1); py = (ny + 1) / 2 * (cfg.input.height - 1)
        sc = b["scale"][i].item(); pad = b["pad"][i]
        out.append([(px - pad[0].item()) / sc, (py - pad[1].item()) / sc])
    return out


def run_epoch(model, loader, criterion, optimizer, device, cfg, train):
    model.train(train)
    tot = kp_s = cls_s = 0.0
    pred_px, gt_px, pred_cls, gt_cls = [], [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in tqdm(loader, leave=False):
            img = b["image"].to(device)
            batch = {"kp_norm": b["kp_norm"].to(device),
                     "heatmap": b["heatmap"].to(device),
                     "cls_idx": b["cls_idx"].to(device)}
            kp_out, cls_logits = model(img)
            loss, kp_l, cls_l, coords = criterion(kp_out, cls_logits, batch)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tot += loss.item(); kp_s += kp_l.item(); cls_s += cls_l.item()
            for i in range(img.size(0)):
                ox, oy = coords_norm_to_orig_px(coords[i:i + 1].detach().cpu(),
                                                cfg.input.width, cfg.input.height,
                                                b["scale"][i].item(),
                                                (b["pad"][i][0].item(), b["pad"][i][1].item()))
                pred_px.append([ox.item(), oy.item()])
            gt_px.extend(_gt_orig_px(b, cfg))
            pred_cls.append(cls_logits.argmax(1).cpu()); gt_cls.append(b["cls_idx"])
    n = len(loader)
    pred_px = torch.tensor(pred_px); gt_px = torch.tensor(gt_px)
    pc = torch.cat(pred_cls); gc = torch.cat(gt_cls)
    return {"loss": tot / n, "kp": kp_s / n, "cls": cls_s / n,
            "pck10": pck(pred_px, gt_px, 10), "pck25": pck(pred_px, gt_px, 25),
            "pck50": pck(pred_px, gt_px, 50), "f1": macro_f1(pc, gc, cfg.model_num_classes)}


def train(config_path):
    cfg = load_config(config_path)
    cfg.model_num_classes = len(cfg.classes)
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    train_ds_full = GCPDataset(cfg.paths.label_file, cfg.paths.train_dir,
                               get_train_transform(cfg.input.width, cfg.input.height),
                               cfg.class_to_idx, cfg.input.width, cfg.input.height, cfg.input.heatmap_stride)
    val_ds_full = GCPDataset(cfg.paths.label_file, cfg.paths.train_dir,
                             get_val_transform(cfg.input.width, cfg.input.height),
                             cfg.class_to_idx, cfg.input.width, cfg.input.height, cfg.input.heatmap_stride)
    split_in = [(g, c) for _, _, _, c, g in train_ds_full.samples]
    tr_idx, va_idx = group_stratified_split(split_in, cfg.training.val_frac, cfg.training.seed)
    train_ds = Subset(train_ds_full, tr_idx)
    val_ds = Subset(val_ds_full, va_idx)

    def dl(ds, sh):
        return DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=sh,
                          num_workers=cfg.training.num_workers, pin_memory=True)
    train_loader, val_loader = dl(train_ds, True), dl(val_ds, False)

    model = GCPModel(len(cfg.classes), cfg.model.pretrained, cfg.model.kp_head,
                     cfg.model.dropout, cfg.input.heatmap_stride).to(device)
    cw = class_weights_from([train_ds_full.samples[i] for i in tr_idx], len(cfg.classes)).to(device)
    criterion = GCPLoss(cfg.training.kp_loss_weight, cfg.training.cls_loss_weight,
                        cfg.training.dsnt_reg_weight, cw, cfg.model.kp_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.num_epochs)

    best, patience, rows = -1.0, 0, []
    for ep in range(1, cfg.training.num_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, opt, device, cfg, True)
        va = run_epoch(model, val_loader, criterion, None, device, cfg, False)
        sched.step()
        score = va["pck10"] + va["f1"]
        rows.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                     **{f"val_{k}": v for k, v in va.items()}})
        print(f"Ep{ep:3d} val loss={va['loss']:.3f} PCK10={va['pck10']:.3f} "
              f"PCK25={va['pck25']:.3f} F1={va['f1']:.3f}")
        if score > best:
            best, patience = score, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "cfg_classes": cfg.classes, "val": va},
                       f"{cfg.paths.output_dir}/best.pt")
            print(f"  -> saved best (score={best:.3f})")
        else:
            patience += 1
            if patience >= cfg.training.patience:
                print(f"early stop @ {ep}"); break
    with open(f"{cfg.paths.output_dir}/training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
