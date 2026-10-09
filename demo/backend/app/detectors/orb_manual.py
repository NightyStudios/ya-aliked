import cv2
import numpy as np

from ..registry import register_detector

from ORB.manual_orb import manual_orb
from ORB.manual_orb_fast import manual_orb_fast
from ORB.manual_orb_pyramid import manual_orb_pyramid


def _run_manual(image_bgr: np.ndarray, **params):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pts, desc = manual_orb(gray, **params)
    return pts, desc, "hamming"


def _run_fast(image_bgr: np.ndarray, **params):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pts, desc = manual_orb_fast(gray, **params)
    return pts, desc, "hamming"


def _run_pyramid(image_bgr: np.ndarray, **params):
    pts, desc = manual_orb_pyramid(image_bgr, **params)
    return pts, desc, "hamming"


register_detector("orb_manual", _run_manual, hidden=True,
    description="Pure-Python manual ORB, single scale (baseline, slow)")

register_detector("orb_fast", _run_fast, hidden=True,
    description="Vectorized manual ORB, single scale")

register_detector("orb_pyramid", _run_pyramid,
    description="Manual ORB with scale pyramid (final)")