import cv2
import numpy as np

from ..registry import register_matcher


_NORM = {"hamming": cv2.NORM_HAMMING, "l2": cv2.NORM_L2}


def bf_match(desc_a: np.ndarray, desc_b: np.ndarray, norm: str,
             cross_check: bool = True, ratio: float | None = None):
    """BF matcher. If `ratio` is set — Lowe's ratio test (knnMatch k=2),
    else — plain match with optional crossCheck."""
    cv_norm = _NORM[norm]

    if ratio is not None:
        bf = cv2.BFMatcher(cv_norm, crossCheck=False)
        knn = bf.knnMatch(desc_a, desc_b, k=2)
        out = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                out.append((int(m.queryIdx), int(m.trainIdx), float(m.distance)))
        return out

    bf = cv2.BFMatcher(cv_norm, crossCheck=cross_check)
    raw = bf.match(desc_a, desc_b)
    return [(int(m.queryIdx), int(m.trainIdx), float(m.distance)) for m in raw]


register_matcher("bf", bf_match)