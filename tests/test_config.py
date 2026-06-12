from src.config import load_config

def test_config_classes_and_maps():
    cfg = load_config("configs/config.yaml")
    assert cfg.classes == ["Cross", "Square", "L-Shape"]
    assert cfg.class_to_idx == {"Cross": 0, "Square": 1, "L-Shape": 2}
    assert cfg.idx_to_class[2] == "L-Shape"
    assert cfg.input.width == 768 and cfg.input.height == 576
    assert cfg.model.kp_head == "heatmap"
