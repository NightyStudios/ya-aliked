"""SIFT-детектор на базе OpenCV (cv2.SIFT_create).

Дескрипторы — float32, 128-мерные, сравниваются по L2 (norm="l2"), поэтому
подходят и JS-матчеру фронта, и бэкендовому bf (cross-check / Lowe ratio).
"""

import cv2
import numpy as np

from ..registry import register_detector


def _run_cv_sift(image_bgr: np.ndarray, n_features: int = 1000,
                 n_octave_layers: int = 3, contrast_threshold: float = 0.04,
                 edge_threshold: float = 10.0, sigma: float = 1.6, **params):
    """(pts (N,2) float32, desc (N,128) float32, "l2")."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=int(n_features),
        nOctaveLayers=int(n_octave_layers),
        contrastThreshold=float(contrast_threshold),
        edgeThreshold=float(edge_threshold),
        sigma=float(sigma),
    )
    kps, desc = sift.detectAndCompute(gray, None)
    if kps:
        pts = np.asarray([kp.pt for kp in kps], dtype=np.float32)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return pts, desc, "l2"


register_detector("sift_cv", _run_cv_sift,
    description="OpenCV SIFT (128-d float32, L2)",
    group="SIFT")
