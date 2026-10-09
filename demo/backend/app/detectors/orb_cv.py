import cv2
import numpy as np

from ..registry import register_detector


def _run_cv_orb(image_bgr: np.ndarray, n_features: int = 1000, **params):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=int(n_features))
    kps, desc = orb.detectAndCompute(gray, None)
    if kps:
        pts = np.asarray([kp.pt for kp in kps], dtype=np.float32)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
    if desc is None:
        desc = np.zeros((0, 32), dtype=np.uint8)
    return pts, desc, "hamming"


register_detector("orb_cv", _run_cv_orb,
    description="OpenCV reference ORB")