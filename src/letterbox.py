import cv2
import numpy as np


def letterbox(img, out_w: int, out_h: int, pad_value: int = 0):
    """Resize preserving aspect, pad to (out_h, out_w). Returns (canvas, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    if img.ndim == 3:
        canvas = np.full((out_h, out_w, img.shape[2]), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((out_h, out_w), pad_value, dtype=img.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def fwd_point(xy, scale, pad):
    """Original-pixel point -> letterboxed-pixel point."""
    return (xy[0] * scale + pad[0], xy[1] * scale + pad[1])


def inv_point(xy, scale, pad):
    """Letterboxed-pixel point -> original-pixel point."""
    return ((xy[0] - pad[0]) / scale, (xy[1] - pad[1]) / scale)
