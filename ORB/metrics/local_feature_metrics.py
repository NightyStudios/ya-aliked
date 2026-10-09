"""Model-independent MMA/MHA/MS evaluation for HPatches full sequences."""
from pathlib import Path
from typing import Protocol
import time

import cv2
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_ROOT / 'data' / 'hpatches-sequences-release'


class DatasetError(ValueError):
    """Dataset is missing, incomplete or has an incompatible layout."""


def dataset_sequences(dataset=None, *, exclude=(), limit_sequences=None):
    """Resolve relative paths against the script directory, never the cwd."""
    path = DEFAULT_DATASET if dataset is None else Path(dataset).expanduser()
    if not path.is_absolute():
        path = SCRIPT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise DatasetError(f'HPatches dataset directory does not exist: {path}. '
                           f'Place the extracted full sequences in {DEFAULT_DATASET}.')
    sequences = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith(('i_', 'v_')))
    if not sequences:
        raise DatasetError(f'No i_* / v_* sequence directories found in {path}. '
                           'Expected the extracted hpatches-sequences-release folder.')
    sequences = [p for p in sequences if p.name not in exclude][:limit_sequences]
    if not sequences:
        raise DatasetError(f'No sequences left to evaluate in {path} after exclusions.')
    for sequence in sequences:
        filenames = [f'{i}.ppm' for i in range(1, 7)] + [f'H_1_{i}' for i in range(2, 7)]
        for filename in filenames:
            if not (sequence / filename).is_file():
                raise DatasetError(f'HPatches dataset is incomplete: missing {sequence / filename}. '
                                   'Expected 1.ppm–6.ppm and H_1_2–H_1_6 in every sequence; '
                                   'the stacked-patch release is incompatible.')
    return path, sequences


class FeatureExtractor(Protocol):
    """BGR uint8 image -> (N×2 pixel coordinates, N×D descriptors).

    Coordinates must refer to the original image, even if inference resizes it.
    Descriptors are float arrays for cosine matching, uint8 for Hamming.
    Empty detections: return an empty (0, 2) array and None descriptors.
    """

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...


def project(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return homogeneous[:, :2] / homogeneous[:, 2:]


def match(descriptors_a, descriptors_b, binary=False):
    if descriptors_a is None or descriptors_b is None or not len(descriptors_a) or not len(descriptors_b):
        return np.empty((0, 2), dtype=int)
    if not binary:
        descriptors_a = descriptors_a.astype(np.float32, copy=True)
        descriptors_b = descriptors_b.astype(np.float32, copy=True)
        descriptors_a /= np.maximum(np.linalg.norm(descriptors_a, axis=1, keepdims=True), 1e-12)
        descriptors_b /= np.maximum(np.linalg.norm(descriptors_b, axis=1, keepdims=True), 1e-12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2, crossCheck=True)
    return np.asarray([(m.queryIdx, m.trainIdx) for m in matcher.match(descriptors_a, descriptors_b)], dtype=int).reshape(-1, 2)


def visible(points, shape):
    h, w = shape[:2]
    return (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0)
            & (points[:, 0] <= w - 1) & (points[:, 1] >= 0)
            & (points[:, 1] <= h - 1))


def pair_metrics(a, b, matches, homography, shape, thresholds, target_shape):
    """Fractions at each pixel threshold; MS uses symmetric co-visible counts."""
    pa, pb = a[matches[:, 0]], b[matches[:, 1]]
    warped_a = project(a, homography)
    warped_b = project(b, np.linalg.inv(homography))
    visible_a = visible(a, shape) & visible(warped_a, target_shape)
    visible_b = visible(b, target_shape) & visible(warped_b, shape)
    errors = np.linalg.norm(project(pa, homography) - pb, axis=1)
    mma = [float(np.mean(errors <= t)) if len(errors) else 0.0 for t in thresholds]
    covisible_matches = visible_a[matches[:, 0]] & visible_b[matches[:, 1]]
    n_a, n_b = int(visible_a.sum()), int(visible_b.sum())
    ms = [float(2 * np.sum((errors <= t) & covisible_matches) / (n_a + n_b))
          if n_a + n_b else 0.0 for t in thresholds]
    predicted = None
    if len(matches) >= 4:
        predicted, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    corner_errors = np.full(4, np.inf)
    if predicted is not None:
        h, w = shape[:2]
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        values = np.linalg.norm(project(corners, homography) - project(corners, predicted), axis=1)
        corner_errors = np.where(np.isfinite(values), values, np.inf)
    return {'mma': mma, 'mha': [float(np.mean(corner_errors <= t)) for t in thresholds],
            'ms': ms, 'covisible_keypoints_a': n_a, 'covisible_keypoints_b': n_b,
            'corner_errors_px': [float(e) if np.isfinite(e) else None for e in corner_errors],
            'matches': len(matches)}


def summarize(rows, thresholds):
    output = {}
    for name, selected in [('all', rows), ('illumination', [r for r in rows if r['sequence'].startswith('i_')]),
                           ('viewpoint', [r for r in rows if r['sequence'].startswith('v_')])]:
        if not selected:
            continue
        output[name] = {'pairs': len(selected), 'mean_matches': float(np.mean([r['matches'] for r in selected]))}
        for metric in ('mma', 'mha', 'ms'):
            for threshold, value in zip(thresholds, np.mean([r[metric] for r in selected], axis=0)):
                output[name][f'{metric.upper()}@{threshold:g}'] = float(value)
    return output


def evaluate(dataset, extract: FeatureExtractor, *, binary=False,
             thresholds=(1, 2, 3, 4, 5), exclude=(), limit_sequences=None):
    """Return JSON-compatible metrics for a supplied extractor.

    MMA: mean per-pair fraction of correct mutual nearest-neighbor matches.
    MHA: mean fraction of four corners with error <= threshold.
    MS: 2 * correct co-visible matches / (co-visible points A + B).
    Failed/empty pairs count as zero. Scores are fractions, not percentages.
    """
    if not thresholds or any(not np.isfinite(t) or t <= 0 for t in thresholds):
        raise ValueError('Thresholds must be finite and positive.')
    if limit_sequences is not None and limit_sequences <= 0:
        raise ValueError('Sequence limit must be positive.')
    dataset_path, sequences = dataset_sequences(dataset, exclude=exclude, limit_sequences=limit_sequences)
    cv2.setRNGSeed(0)
    rows, counts, timings = [], [], []
    for sequence in sequences:
        features = []
        shapes = []
        ref_shape = None
        for index in range(1, 7):
            image = cv2.imread(str(sequence / f'{index}.ppm'))
            if image is None:
                raise ValueError(f'Cannot read {sequence / f"{index}.ppm"}')
            shapes.append(image.shape)
            if index == 1:
                ref_shape = image.shape
            start = time.perf_counter()
            points, descriptors = extract(image)
            timings.append(time.perf_counter() - start)
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('Extractor must return finite keypoints with shape (N, 2).')
            if descriptors is None:
                if len(points):
                    raise ValueError('Nonempty keypoints require descriptors.')
            else:
                descriptors = np.asarray(descriptors)
                if descriptors.ndim != 2 or len(descriptors) != len(points) or not np.isfinite(descriptors).all():
                    raise ValueError('Descriptors must be finite (N, D), one per keypoint.')
                if binary and descriptors.dtype != np.uint8:
                    raise ValueError('Hamming descriptors must have dtype uint8.')
            features.append((points, descriptors))
            counts.append(len(points))
        for index in range(2, 7):
            a, da = features[0]
            b, db = features[index - 1]
            h = np.loadtxt(sequence / f'H_1_{index}')
            if h.shape != (3, 3) or not np.isfinite(h).all():
                raise ValueError(f'Invalid homography in {sequence}')
            result = pair_metrics(a, b, match(da, db, binary), h, ref_shape, thresholds, shapes[index - 1])
            rows.append(dict(sequence=sequence.name, target=index, **result))
    return {'dataset': str(dataset_path), 'summary': summarize(rows, thresholds), 'thresholds_px': list(thresholds),
            'binary': binary, 'metric_version': 2,
            'ms_denominator': '(covisible_a + covisible_b) / 2',
            'mha_definition': 'fraction of correct corners', 'mean_keypoints': float(np.mean(counts)),
            'mean_extraction_ms': float(np.mean(timings) * 1000), 'pairs': rows}
