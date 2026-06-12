# GCP Pose Estimation — Skylark Drones Assignment

A dual-head CNN that, given a full-resolution aerial drone image, predicts the **(x, y) pixel center** of a Ground Control Point (GCP) marker and classifies its **shape** as `Cross`, `Square`, or `L-Shape`.

## Architecture

**Encoder:** ResNet-50 (ImageNet-pretrained). The spatial feature map is *kept* (no global pooling before the keypoint head), which preserves the localization detail that PCK@10 needs.

**Keypoint head — DSNT / soft-argmax:** a light decoder produces a single-channel heatmap at stride 4; a differentiable soft-argmax reads off a sub-pixel (x, y). Trained on a coordinate (MSE) loss plus a heatmap-distribution regularizer. This beats direct coordinate regression for tight-threshold PCK.

**Classification head:** global-pooled encoder bottleneck → dropout → linear(3).

A `regression` keypoint head (linear→sigmoid + Wing-style L1) is available via `model.kp_head` as a fallback/ablation; default is `heatmap`.

## Key data-handling decisions (driven by the real labels)

The production labels (`gcp_marks.json`) forced four choices that a naive pipeline gets wrong:

1. **Class is `L-Shape`** (not "L-Shaped").
2. **Image resolution varies** across drones/sensors (marks reach ~3940×2915). Each image's true H×W is read at load time; keypoints are normalized/de-normalized and **PCK is scored in each image's own pixel space** — nothing is hardcoded.
3. **Some entries lack `verified_shape`** — imputed from the GCP folder's majority class (dropped only if the whole folder is unlabelled).
4. **Heavy near-duplicate frames** of the same physical GCP. A random split would leak them across train/val. We use a **group-stratified split** (by GCP folder, stratified by class) so no marker appears in both splits.

## Training strategy

- **Letterbox** to 768×576 (preserve aspect, pad) — one documented coordinate convention reused by dataset, training, metrics, and inference.
- **Augmentation** (keypoint-aware, Albumentations): H/V flips, ±180° rotation, scale jitter, color jitter, Gaussian noise, motion blur, random shadow.
- **Loss:** DSNT keypoint loss + inverse-frequency **weighted cross-entropy** (class imbalance).
- **Optim:** AdamW + CosineAnnealingLR, gradient clipping, early stopping on `PCK@10 + Macro-F1`, best checkpoint saved.
- **Inference:** flip TTA (identity + h-flip + v-flip; predictions un-flipped and averaged), then letterbox-inverted to original pixels.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# CPU torch is fine for tests/inference; Kaggle provides the GPU for training
pip install -r requirements.txt
```

Dataset layout expected (see `configs/config.yaml` to adjust paths):

```
data/
  gcp_marks.json        # labels
  train_dataset/        # nested Site/Survey/GCPxx/*.JPG
  test_dataset/         # unlabelled images
```

## Run

```bash
# Train (GPU strongly recommended — use Kaggle; see notebooks/kaggle_train.ipynb)
python train.py --config configs/config.yaml

# Inference -> predictions.json (mirrors the label schema exactly)
python predict.py --config configs/config.yaml --checkpoint outputs/best.pt --output predictions.json
```

`predictions.json` format:

```json
{ "Site/Survey/GCPxx/IMG.JPG": { "mark": {"x": 1024.5, "y": 680.2}, "verified_shape": "Cross" } }
```

## Tests

```bash
python -m pytest -q     # 17 unit tests: config, letterbox round-trip, no-leakage split,
                        # dataset contract, model shapes, soft-argmax, losses, metrics, predict schema
```

## Notes on compute

Training is done on **Kaggle** (free P100/T4). A typical full run is ~1.5–3 GPU-hours. The repo's unit tests and inference run on CPU; full-model training does not fit on a low-RAM laptop.

## Results

| Metric | Value |
|--------|-------|
| PCK@10 | _fill from outputs/training_log.csv_ |
| PCK@25 | _fill_ |
| PCK@50 | _fill_ |
| Macro-F1 | _fill_ |

## Model weights

`outputs/best.pt` — _add shareable Drive link after training._
