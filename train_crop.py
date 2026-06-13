import argparse
from src.train_crop import train_crop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    train_crop(ap.parse_args().config)
