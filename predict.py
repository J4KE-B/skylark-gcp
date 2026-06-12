import argparse
from src.predict import predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--output", default="predictions.json")
    a = ap.parse_args()
    predict(a.config, a.checkpoint, a.output)
