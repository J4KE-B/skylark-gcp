import numpy as np
from src.letterbox import letterbox, fwd_point, inv_point

def test_letterbox_shape_and_roundtrip():
    img = np.full((300, 400, 3), 128, dtype=np.uint8)  # H=300,W=400
    out, scale, pad = letterbox(img, 768, 576)
    assert out.shape == (576, 768, 3)
    # a known point maps forward then inverts back to itself
    p = (123.4, 222.2)
    fp = fwd_point(p, scale, pad)
    ip = inv_point(fp, scale, pad)
    assert abs(ip[0] - p[0]) < 1e-3 and abs(ip[1] - p[1]) < 1e-3

def test_letterbox_preserves_aspect():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    _, scale, pad = letterbox(img, 768, 576)
    # 400*scale <= 768 and 300*scale <= 576, and one dimension is filled
    assert round(400 * scale) <= 768 and round(300 * scale) <= 576
    assert pad[0] == 0 or pad[1] == 0  # padding only on one axis
