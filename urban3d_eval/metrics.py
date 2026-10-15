"""MMA / MHA against Urban3D dense ground truth.

Same definitions as ORB/metrics/local_feature_metrics.py (HPatches), with the
ground truth taken from the dataset's dense pixel correspondences instead of
a single homography:

  MMA@t  fraction of matches whose A-point, moved to B by the GT, lands within
         t px of the matched B-point. Matches whose A-point has no GT
         (NoData / outside the frame / occluded in B) are not scored.
  MHA@t  fraction of the 4 corners of A whose position under the homography
         estimated from the matches (RANSAC, 3 px) is within t px of the
         position under the reference homography.
  The reference homography is fitted by RANSAC to the dense GT correspondences
  (the dominant plane of the scene; the scene is 3D, so no exact one exists).

Pairs with too few matches score zero, as in the HPatches module.
"""
import cv2
import numpy as np

from ORB.metrics.local_feature_metrics import corner_errors, estimate_homography
from urban3d_eval.dataset import SOURCE_VISIBLE

REFERENCE_STEP = 8          # GT grid step (px) for the reference homography
REFERENCE_THRESHOLD = 3.0   # RANSAC threshold (px), same as for predicted H


def sample_forward(gt, points):
    """GT position in B of fractional A points (N, 2) given as [col, row].

    Bilinear interpolation of forward_xy over the visible source pixels
    (weights renormalised); points whose nearest pixel is not visible are
    invalid. Returns ((N, 2) float64, (N,) bool).
    """
    h, w = gt.source_status.shape
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)

    nearest = gt.source_status[np.rint(y).astype(int), np.rint(x).astype(int)] == SOURCE_VISIBLE
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    total = np.zeros(len(points))
    value = np.zeros((len(points), 2))
    for dx, dy, weight in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        ix, iy = np.minimum(x0 + dx, w - 1), np.minimum(y0 + dy, h - 1)
        weight = weight * (gt.source_status[iy, ix] == SOURCE_VISIBLE)
        total += weight
        value += weight[:, None] * np.nan_to_num(gt.forward_xy[iy, ix].astype(np.float64))
    valid = inside & nearest & (total > 1e-9)
    value[valid] /= total[valid, None]
    return value, valid


def reference_homography(gt, step=REFERENCE_STEP, threshold=REFERENCE_THRESHOLD):
    """(H or None, inlier ratio) fitted to the visible GT correspondences."""
    h, w = gt.source_status.shape
    rows, cols = np.mgrid[0:h:step, 0:w:step]
    keep = gt.source_status[rows, cols] == SOURCE_VISIBLE
    src = np.column_stack((cols[keep], rows[keep])).astype(np.float64)
    dst = gt.forward_xy[rows[keep], cols[keep]].astype(np.float64)
    if len(src) < 4:
        return None, 0.0
    cv2.setRNGSeed(0)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold,
                                          maxIters=5000, confidence=0.999)
    if homography is None:
        return None, 0.0
    return homography, float(mask.mean())


def pair_metrics(points_a, points_b, matches, gt, reference, shape, thresholds):
    """Per-pair MMA / MHA at every threshold.

    points_*: (N, 2) keypoints [col, row]; matches: (M, 2) indices into them;
    reference: reference homography A -> B (or None); shape: shape of A.
    """
    pa, pb = points_a[matches[:, 0]], points_b[matches[:, 1]]
    projected, valid = sample_forward(gt, pa)
    errors = np.linalg.norm(projected[valid] - pb[valid], axis=1)
    mma = [float(np.mean(errors <= t)) if len(errors) else 0.0 for t in thresholds]

    errors_c = np.full(4, np.inf)
    if reference is not None:
        errors_c = corner_errors(shape, reference, estimate_homography(pa, pb))
    mha = [float(np.mean(errors_c <= t)) for t in thresholds]
    return {
        'mma': mma, 'mha': mha,
        'matches': int(len(matches)), 'scored_matches': int(valid.sum()),
        'median_error_px': float(np.median(errors)) if len(errors) else None,
        'corner_errors_px': [float(e) if np.isfinite(e) else None for e in errors_c],
    }
