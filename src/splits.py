import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def group_stratified_split(samples, val_frac: float = 0.15, seed: int = 42):
    """samples: list of (group, class_idx). Returns (train_idx, val_idx) numpy arrays.
    Groups never span both splits; split is stratified by class."""
    groups = [g for g, _ in samples]
    y = np.array([c for _, c in samples])
    n_splits = max(2, round(1.0 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, val_idx
