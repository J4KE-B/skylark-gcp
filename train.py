import argparse
from src.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    train(ap.parse_args().config)
