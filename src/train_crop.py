import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.crop_dataset import GCPCropDataset, build_samples, crop_train_tf, crop_val_tf
from src.crop_model import CropClassifier
from src.metrics import macro_f1
from src.splits import group_stratified_split
from src.train import class_weights_from, set_seed


def _run(model, loader, criterion, optimizer, device, n_cls, train):
    model.train(train)
    tot = 0.0
    preds, gts = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in tqdm(loader, leave=False):
            x = b["image"].to(device)
            y = b["cls_idx"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item()
            preds.append(logits.argmax(1).cpu()); gts.append(b["cls_idx"])
    return tot / len(loader), macro_f1(torch.cat(preds), torch.cat(gts), n_cls)


def train_crop(config_path):
    cfg = load_config(config_path)
    n_cls = len(cfg.classes)
    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    samples = build_samples(cfg.paths.label_file, cfg.paths.train_dir, cfg.class_to_idx)
    # Split by SITE (first path component), not GCP folder: shape classification must generalize
    # to unseen sites. A per-GCP-folder split leaves the same sites in train and val, so the
    # classifier scores ~perfectly by memorizing site/terrain and then collapses on new sites.
    site_groups = [(rel.split("/")[0], c) for rel, _, _, c, _ in samples]
    tr_idx, va_idx = group_stratified_split(site_groups, cfg.training.val_frac, cfg.training.seed)
    tr_samples = [samples[i] for i in tr_idx]
    va_samples = [samples[i] for i in va_idx]
    n_tr_sites = len({samples[i][0].split('/')[0] for i in tr_idx})
    n_va_sites = len({samples[i][0].split('/')[0] for i in va_idx})
    print(f"crop train/val = {len(tr_samples)}/{len(va_samples)} "
          f"({n_tr_sites} train sites, {n_va_sites} val sites, no overlap)")

    train_ds = GCPCropDataset(tr_samples, cfg.paths.train_dir, crop_train_tf(), train=True)
    val_ds = GCPCropDataset(va_samples, cfg.paths.train_dir, crop_val_tf(), train=False)
    bs = getattr(cfg.training, "crop_batch_size", 32)

    def dl(ds, sh):
        return DataLoader(ds, batch_size=bs, shuffle=sh,
                          num_workers=cfg.training.num_workers, pin_memory=True)
    tl, vl = dl(train_ds, True), dl(val_ds, False)

    model = CropClassifier(n_cls, pretrained=True, dropout=cfg.model.dropout).to(device)
    cw = class_weights_from(tr_samples, n_cls).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    epochs = getattr(cfg.training, "crop_epochs", 30)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, patience, rows = -1.0, 0, []
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1 = _run(model, tl, criterion, opt, device, n_cls, True)
        va_loss, va_f1 = _run(model, vl, criterion, None, device, n_cls, False)
        sched.step()
        rows.append({"epoch": ep, "train_loss": tr_loss, "train_f1": tr_f1,
                     "val_loss": va_loss, "val_f1": va_f1})
        print(f"Ep{ep:3d} train_loss={tr_loss:.3f} f1={tr_f1:.3f} | "
              f"val_loss={va_loss:.3f} f1={va_f1:.3f}")
        if va_f1 > best:
            best, patience = va_f1, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "cfg_classes": cfg.classes, "val_f1": va_f1},
                       f"{cfg.paths.output_dir}/best_crop.pt")
            print(f"  -> saved best_crop (val_f1={best:.3f})")
        else:
            patience += 1
            if patience >= cfg.training.patience:
                print(f"early stop @ {ep}"); break
    with open(f"{cfg.paths.output_dir}/crop_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
