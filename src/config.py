import yaml
from types import SimpleNamespace


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    cfg.class_to_idx = {c: i for i, c in enumerate(cfg.classes)}
    cfg.idx_to_class = {i: c for i, c in enumerate(cfg.classes)}
    return cfg
