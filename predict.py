import argparse
from src.predict import predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--output", default="predictions.json")
    ap.add_argument("--crop-checkpoint", default=None,
                    help="optional ResNet-18 crop classifier; overrides shapes via high-res crops")
    a = ap.parse_args()
    predict(a.config, a.checkpoint, a.output, a.crop_checkpoint)
