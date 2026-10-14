"""RANSAC-фильтр матчей по гомографии (пост-фильтр /api/pipeline)."""

import cv2
import numpy as np


def ransac_filter_matches(raw, pts_a, pts_b, *,
                          thresh=3.0, confidence=0.999, max_iters=5000):
    """Оставляет только инлайеры гомографии RANSAC.

    raw : list[(a_idx, b_idx, dist)]
    pts_a/pts_b : (N,2) координаты ключевых точек.
    Возвращает новый список; при <4 матчах или ошибке поиска гомографии
    возвращает raw без изменений.
    """
    if len(raw) < 4:
        return raw

    idx_a = np.array([a for a, _, _ in raw], dtype=np.int32)
    idx_b = np.array([b for _, b, _ in raw], dtype=np.int32)
    p0 = np.asarray(pts_a, dtype=np.float32)[idx_a]
    p1 = np.asarray(pts_b, dtype=np.float32)[idx_b]

    _, mask = cv2.findHomography(
        p0, p1, method=cv2.RANSAC,
        ransacReprojThreshold=thresh,
        maxIters=max_iters, confidence=confidence,
    )
    if mask is None:
        return raw
    keep = mask.ravel().astype(bool)
    return [m for m, k in zip(raw, keep) if k]
