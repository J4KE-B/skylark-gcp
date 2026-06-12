from src.splits import group_stratified_split


def test_no_group_leakage_and_all_classes_in_val():
    # 15 groups, 5 per class; each group has several near-duplicate frames.
    # (Enough groups per class that a stratified fold contains all 3 classes,
    #  mirroring the real dataset where every class spans many GCP folders.)
    samples = []
    for g in range(15):
        cls = g % 3
        for _ in range(5):
            samples.append((f"site/GCP{g}", cls))
    train_idx, val_idx = group_stratified_split(samples, val_frac=0.2, seed=42)
    train_groups = {samples[i][0] for i in train_idx}
    val_groups = {samples[i][0] for i in val_idx}
    assert train_groups.isdisjoint(val_groups)            # no leakage
    assert len(set(samples[i][1] for i in val_idx)) == 3  # all classes in val
    assert len(train_idx) + len(val_idx) == len(samples)
