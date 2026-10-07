"""MMA, both MHA definitions, and symmetric co-visible MS."""
import cv2
import numpy as np


def project(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return homogeneous[:, :2] / homogeneous[:, 2:]



def visible(points, shape):
    h, w = shape[:2]
    return (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0)
            & (points[:, 0] <= w - 1) & (points[:, 1] >= 0)
            & (points[:, 1] <= h - 1))



def metrics(a, b, matches, h, shape, target_shape):
    pa, pb = a[matches[:, 0]], b[matches[:, 1]]
    error = np.linalg.norm(project(pa, h) - pb, axis=1)
    visible_a = visible(a, shape) & visible(project(a, h), target_shape)
    visible_b = visible(b, target_shape) & visible(project(b, np.linalg.inv(h)), shape)
    covisible_matches = visible_a[matches[:, 0]] & visible_b[matches[:, 1]]
    denominator = int(visible_a.sum()) + int(visible_b.sum())
    predicted = None
    if len(matches) >= 4:
        predicted, _ = cv2.findHomography(pa, pb, cv2.RANSAC)
    corners_error = np.full(4, np.inf)
    if predicted is not None:
        height, width = shape[:2]
        corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
        corners_error = np.linalg.norm(project(corners, h) - project(corners, predicted), axis=1)
    return {
        'mma': [float(np.mean(error <= t)) if len(error) else 0.0 for t in (1, 2, 3)],
        'ms': [float(2 * np.sum((error <= t) & covisible_matches) / denominator)
               if denominator else 0.0 for t in (1, 2, 3)],
        'mha_author_code': [float(np.mean(corners_error) <= t) for t in (1, 2, 3)],
        'mha_corner_fraction': [float(np.mean(corners_error <= t)) for t in (1, 2, 3)],
        'matches': len(matches),
    }

