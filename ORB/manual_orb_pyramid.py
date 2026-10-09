"""Manual ORB with scale pyramid.

Adds scale invariance by:
1. Building a pyramid of progressively downscaled images.
2. Running FAST + Harris + NMS on each level.
3. Keeping a per-level quota of features (as in OpenCV ORB).
4. Computing orientation and rBRIEF descriptors on the level where
   each keypoint was detected.
5. Scaling keypoint coordinates back to level 0.

Reuses building blocks from manual_orb.py and manual_orb_fast.py.
"""
from typing import List, Tuple

import cv2
import numpy as np

from Красота.backend.ORB.manual_orb import (
    fast_detect_cv, fast_detect_manual,
    intensity_centroid_orientation,
    sample_pairs_gaussian_gii,
)
from Красота.backend.ORB.manual_orb_fast import (
    _nms_fast, _harris_scores_fast,
    _precompute_rotations, _angle_to_bucket,
    _brief_descriptor_vec,
)


DEFAULT_PAIRS_A, DEFAULT_PAIRS_B = sample_pairs_gaussian_gii(256, 31, seed=42)


# ---------------------------------------------------------------------------
# Pyramid
# ---------------------------------------------------------------------------

def build_pyramid(gray: np.ndarray, n_levels: int = 8,
                  scale_factor: float = 1.2
                  ) -> List[Tuple[np.ndarray, float]]:
    """Returns [(level_image, scale_to_original), ...].

    Level 0: original, scale = 1.0
    Level k: downscaled by scale_factor^k, scale = scale_factor^k
    """
    h, w = gray.shape
    pyramid = [(gray, 1.0)]
    for level in range(1, n_levels):
        scale = scale_factor ** level
        new_w = max(int(round(w / scale)), 8)
        new_h = max(int(round(h / scale)), 8)
        img_lvl = cv2.resize(gray, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        pyramid.append((img_lvl, scale))
    return pyramid


def features_per_level(n_features: int, n_levels: int = 8,
                       scale_factor: float = 1.2) -> List[int]:
    """Per-level quota from OpenCV ORB.

    Level 0 gets the most features; each next level gets fewer
    by a factor of 1/scale_factor.
    """
    factor = 1.0 / scale_factor
    total = sum(factor ** k for k in range(n_levels))
    quotas = []
    for level in range(n_levels):
        q = int(round(n_features * (factor ** level) / total))
        quotas.append(max(q, 0))
    return quotas


# ---------------------------------------------------------------------------
# Full pyramid ORB
# ---------------------------------------------------------------------------

def manual_orb_pyramid(image_bgr: np.ndarray,
                       n_features: int = 1000,
                       fast_threshold: int = 20,
                       n_levels: int = 8,
                       scale_factor: float = 1.2,
                       patch_size: int = 31,
                       subwindow: int = 5,
                       use_orientation: bool = True,
                       nms_radius: int = 3,
                       use_cv_fast: bool = True,
                       discretize_angles: bool = True,
                       angle_buckets: int = 30,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Manual ORB with pyramid.

    Returns
    -------
    points : (N, 2) float32, coordinates in level-0 (original) image
    descs  : (N, 32) uint8, packed 256-bit rBRIEF descriptors
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pyramid = build_pyramid(gray, n_levels, scale_factor)
    quotas = features_per_level(n_features, n_levels, scale_factor)

    sw = subwindow // 2
    half = patch_size // 2

    rotations = (_precompute_rotations(DEFAULT_PAIRS_A, DEFAULT_PAIRS_B,
                                       angle_buckets)
                 if (use_orientation and discretize_angles) else None)

    pts_out: List[Tuple[float, float]] = []
    descs_out: List[np.ndarray] = []

    for level, ((lvl_img, scale), quota) in enumerate(zip(pyramid, quotas)):
        if quota <= 0:
            continue
        h, w = lvl_img.shape

        # 1. FAST candidates on this level
        if use_cv_fast:
            candidates = fast_detect_cv(lvl_img, threshold=fast_threshold)
        else:
            candidates = fast_detect_manual(lvl_img,
                                            threshold=fast_threshold, n=9)
        if not candidates:
            continue

        # 2. Harris score
        scores = _harris_scores_fast(lvl_img, candidates)

        # 3. NMS + top-quota
        kept, _ = _nms_fast(candidates, scores,
                            radius=nms_radius, max_keep=quota)
        if not kept:
            continue

        # 4. Integral image for vectorized BRIEF
        integral = cv2.integral(lvl_img).astype(np.float32)

        # 5. For each keypoint: orientation + descriptor on level image
        for (x_lvl, y_lvl) in kept:
            if not (half < x_lvl < w - half and half < y_lvl < h - half):
                continue

            if use_orientation:
                theta = intensity_centroid_orientation(lvl_img,
                                                       x_lvl, y_lvl,
                                                       radius=15)
                if discretize_angles:
                    a_rot, b_rot = rotations[_angle_to_bucket(theta,
                                                              angle_buckets)]
                else:
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, -s], [s, c]])
                    a_rot = DEFAULT_PAIRS_A @ R.T
                    b_rot = DEFAULT_PAIRS_B @ R.T
            else:
                a_rot, b_rot = DEFAULT_PAIRS_A, DEFAULT_PAIRS_B

            d = _brief_descriptor_vec(integral, w, h,
                                      x_lvl, y_lvl,
                                      a_rot, b_rot, sw)
            descs_out.append(np.packbits(d))
            pts_out.append((float(x_lvl * scale), float(y_lvl * scale)))

    if not pts_out:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 32), dtype=np.uint8))

    return (np.asarray(pts_out, dtype=np.float32),
            np.asarray(descs_out, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Adapter for metrics.local_feature_metrics.evaluate()
# ---------------------------------------------------------------------------

def manual_orb_extractor_pyramid(n_features: int = 1000,
                                 fast_threshold: int = 20,
                                 n_levels: int = 8,
                                 scale_factor: float = 1.2,
                                 use_orientation: bool = True,
                                 use_cv_fast: bool = True,
                                 discretize_angles: bool = True):
    """Returns a callable matching FeatureExtractor protocol.

    Input:  BGR uint8 image
    Output: (points (N,2) float32, descriptors (N,32) uint8)
    """
    def extract(image_bgr: np.ndarray):
        return manual_orb_pyramid(
            image_bgr,
            n_features=n_features,
            fast_threshold=fast_threshold,
            n_levels=n_levels,
            scale_factor=scale_factor,
            use_orientation=use_orientation,
            use_cv_fast=use_cv_fast,
            discretize_angles=discretize_angles,
        )
    return extract