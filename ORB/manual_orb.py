"""Manual ORB implementation for evaluation with metrics/local_feature_metrics.py.

Provides:
- All FAST/Harris/orientation/rBRIEF building blocks.
- manual_orb_extractor(image_bgr) -> (points Nx2 float32, descriptors Nx32 uint8)
  which is the interface expected by evaluate().
- use_cv_fast=True to swap manual Python FAST detection for the OpenCV C++ one
  (much faster; the rest of the pipeline stays manual).
"""
from typing import List, Tuple, Optional

import cv2
import numpy as np


# 16 pixel offsets on the radius-3 circle (clockwise from top)
CIRCLE_OFFSETS: List[Tuple[int, int]] = [
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3),
]
# High-speed test indices: pixels 1, 5, 9, 13 → 0-based 0, 4, 8, 12
HIGHSPEED_INDICES = [0, 4, 8, 12]


# ---------------------------------------------------------------------------
# FAST
# ---------------------------------------------------------------------------

def fast_segment_states(center: int, values: np.ndarray, threshold: int) -> np.ndarray:
    states = np.zeros(16, dtype=np.int8)
    states[values > center + threshold] = 1
    states[values < center - threshold] = -1
    return states


def has_n_consecutive(states: np.ndarray, n: int = 9) -> bool:
    doubled = np.concatenate([states, states])
    for start in range(16):
        seg = doubled[start:start + n]
        if np.all(seg == 1) or np.all(seg == -1):
            return True
    return False


def fast_segment_test(img: np.ndarray, x: int, y: int,
                      threshold: int = 20, n: int = 9) -> bool:
    h, w = img.shape
    if x < 3 or y < 3 or x >= w - 3 or y >= h - 3:
        return False
    center = int(img[y, x])
    values = np.array([int(img[y + dy, x + dx]) for dx, dy in CIRCLE_OFFSETS])
    return has_n_consecutive(fast_segment_states(center, values, threshold), n)


def fast_detect_manual(img: np.ndarray, threshold: int = 20,
                       n: int = 9) -> List[Tuple[int, int]]:
    """Pure-Python FAST. Slow but is exactly what the notebook demonstrates."""
    h, w = img.shape
    keypoints = []
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            if fast_segment_test(img, x, y, threshold, n):
                keypoints.append((x, y))
    return keypoints


def fast_detect_cv(img: np.ndarray, threshold: int = 20) -> List[Tuple[int, int]]:
    """OpenCV FAST — same segment test, C++ implementation."""
    detector = cv2.FastFeatureDetector_create(threshold=threshold,
                                              nonmaxSuppression=False,
                                              type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kps = detector.detect(img, None)
    return [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]


# ---------------------------------------------------------------------------
# Harris
# ---------------------------------------------------------------------------

def harris_response_manual(img: np.ndarray, k: float = 0.04,
                           window_size: int = 5) -> np.ndarray:
    img_f = img.astype(np.float32)
    Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    Sxx = cv2.GaussianBlur(Ix * Ix, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iy * Iy, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ix * Iy, (window_size, window_size), 0)
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    return det - k * trace * trace


# ---------------------------------------------------------------------------
# Intensity centroid orientation
# ---------------------------------------------------------------------------

def intensity_centroid_orientation(img: np.ndarray, x: int, y: int,
                                   radius: int = 15) -> float:
    h, w = img.shape
    if not (0 <= x < w and 0 <= y < h):
        return 0.0
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx = xx - x
    yy = yy - y
    mask = (xx * xx + yy * yy) <= radius * radius
    I = img[y0:y1, x0:x1].astype(np.float32) * mask
    m00 = I.sum()
    if m00 < 1e-6:
        return 0.0
    m10 = (xx * I).sum()
    m01 = (yy * I).sum()
    return float(np.arctan2(m01, m10))


# ---------------------------------------------------------------------------
# BRIEF / rBRIEF descriptor
# ---------------------------------------------------------------------------

def sample_pairs_gaussian_gii(n_pairs: int = 256, patch_size: int = 31,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """G II from the BRIEF paper: sigma = S/5, clamped to patch bounds."""
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    sigma = patch_size / 5.0
    pts = rng.normal(0, sigma, size=(2 * n_pairs, 2))
    max_coord = half - 2
    pts = np.clip(pts, -max_coord, max_coord)
    return pts[:n_pairs].astype(np.float32), pts[n_pairs:].astype(np.float32)


# Pre-generate the same 256 pairs every call
DEFAULT_PAIRS_A, DEFAULT_PAIRS_B = sample_pairs_gaussian_gii(256, 31, seed=42)


def rotate_pairs(pairs_a: np.ndarray, pairs_b: np.ndarray,
                 theta: float) -> Tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pairs_a @ R.T, pairs_b @ R.T


def brief_descriptor(img: np.ndarray, x: int, y: int,
                     pairs_a: np.ndarray, pairs_b: np.ndarray,
                     subwindow: int = 5) -> np.ndarray:
    """256 binary tests on 5x5 sub-windows around each sampled point."""
    h, w = img.shape
    sw = subwindow // 2
    desc = np.zeros(len(pairs_a), dtype=np.uint8)
    for i, (a, b) in enumerate(zip(pairs_a, pairs_b)):
        ax, ay = int(round(x + a[0])), int(round(y + a[1]))
        bx, by = int(round(x + b[0])), int(round(y + b[1]))
        if not (sw <= ax < w - sw and sw <= ay < h - sw):
            continue
        if not (sw <= bx < w - sw and sw <= by < h - sw):
            continue
        va = img[ay - sw:ay + sw + 1, ax - sw:ax + sw + 1].mean()
        vb = img[by - sw:by + sw + 1, bx - sw:bx + sw + 1].mean()
        desc[i] = 1 if va < vb else 0
    return desc


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms_keypoints(keypoints: List[Tuple[int, int]], scores: np.ndarray,
                  radius: int = 3) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    order = np.argsort(scores)[::-1]
    kept, kept_scores = [], []
    r2 = radius * radius
    for i in order:
        x, y = keypoints[i]
        too_close = False
        for (kx, ky) in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 < r2:
                too_close = True
                break
        if not too_close:
            kept.append((x, y))
            kept_scores.append(scores[i])
    return kept, np.array(kept_scores)


# ---------------------------------------------------------------------------
# Full manual ORB
# ---------------------------------------------------------------------------

def manual_orb(gray: np.ndarray, n_features: int = 1000,
               fast_threshold: int = 20, n: int = 9,
               patch_size: int = 31, subwindow: int = 5,
               use_orientation: bool = True, nms_radius: int = 3,
               use_cv_fast: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Full manual ORB pipeline.

    Returns:
        points: (N, 2) float32 keypoint coordinates on the original image
        descs:  (N, 32) uint8 packed binary descriptors (256 bits each)
    """
    # 1. FAST candidates
    if use_cv_fast:
        candidates = fast_detect_cv(gray, threshold=fast_threshold)
    else:
        candidates = fast_detect_manual(gray, threshold=fast_threshold, n=n)
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

    # 2. Harris scores
    R = harris_response_manual(gray)
    scores = np.array([R[y, x] for (x, y) in candidates])

    # 3. NMS
    kept, kept_scores = nms_keypoints(candidates, scores, radius=nms_radius)
    if len(kept) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

    # 4. Top-N by Harris
    order = np.argsort(kept_scores)[::-1][:n_features]
    kp_final = [kept[i] for i in order]

    # 5. Descriptors
    h, w = gray.shape
    half = patch_size // 2
    pts_out, descs_out = [], []

    for (x, y) in kp_final:
        if not (half < x < w - half and half < y < h - half):
            continue
        if use_orientation:
            theta = intensity_centroid_orientation(gray, x, y, radius=15)
            a_rot, b_rot = rotate_pairs(DEFAULT_PAIRS_A, DEFAULT_PAIRS_B, theta)
        else:
            a_rot, b_rot = DEFAULT_PAIRS_A, DEFAULT_PAIRS_B
        d = brief_descriptor(gray, x, y, a_rot, b_rot, subwindow)
        descs_out.append(np.packbits(d))
        pts_out.append((float(x), float(y)))

    if len(pts_out) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

    return np.asarray(pts_out, dtype=np.float32), np.asarray(descs_out, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Adapter for metrics/local_feature_metrics.evaluate()
# ---------------------------------------------------------------------------

def manual_orb_extractor(n_features: int = 1000,
                         fast_threshold: int = 20,
                         use_orientation: bool = True,
                         use_cv_fast: bool = True):
    """Returns a callable matching the FeatureExtractor protocol.

    Input:  BGR uint8 image (as loaded by cv2.imread).
    Output: (points (N,2) float32, descriptors (N,32) uint8).
    """
    def extract(image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        pts, desc = manual_orb(gray, n_features=n_features,
                               fast_threshold=fast_threshold,
                               use_orientation=use_orientation,
                               use_cv_fast=use_cv_fast)
        return pts, desc
    return extract