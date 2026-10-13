"""Exact class exclusion with a margin measured in original-image pixels."""

import base64

import cv2
import numpy as np
from scipy.spatial import cKDTree


def decode_class_map(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Expected a grayscale uint8 class-map PNG")
    return mask


def filter_by_mask(class_map, points, excluded_classes, margin=0):
    """Keep allowed points, or excluded points within margin of allowed pixels.

    Ported from output/satellite-segmentation/keypoint_filter.py. Class lookup
    rounds to the nearest pixel (ties up); distance uses fractional coordinates.
    The image exterior does not count as an allowed class.
    """
    mask = np.asarray(class_map)
    if mask.ndim != 2 or not mask.size or mask.dtype.kind not in "iu":
        raise ValueError("Expected a nonempty 2D integer class map")
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h, w = mask.shape
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [w, h]):
        raise ValueError("Points must be finite and inside the original image")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and nonnegative")
    if any(c < 0 or c > 255 for c in excluded_classes):
        raise ValueError("Class IDs must be between 0 and 255")
    pixels = np.clip(np.floor(xy + .5).astype(np.int64), [0, 0], [w-1, h-1])
    excluded = np.isin(mask[pixels[:, 1], pixels[:, 0]], excluded_classes)
    keep = ~excluded
    saved = np.zeros(len(xy), dtype=bool)
    if margin > 0 and excluded.any():
        yy, xx = np.nonzero(~np.isin(mask, excluded_classes))
        if len(xx):
            distances = cKDTree(np.column_stack((xx, yy))).query(xy[excluded])[0]
            saved[excluded] = distances <= margin
            keep |= saved
    return {"keep": keep.tolist(), "saved_by_margin": saved.tolist()}
