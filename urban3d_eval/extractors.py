"""Keypoint extractors for the Urban3D benchmark.

Every method is a callable  BGR uint8 image -> (points (N,2) float32,
descriptors (N,D) uint8|float32, norm "hamming"|"l2"), i.e. the detector
contract of demo/backend/app/registry.py.

* ORB / SIFT / XFeat are taken from the demo registry — one implementation
  for the demo and for the benchmark.
* ALIKE lives on the FIR branch (alike.py, alnet.py, soft_detect.py and the
  weights in models/), it is not part of the demo registry.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_BACKEND = REPO_ROOT / 'demo' / 'backend'

# method -> keypoint-budget parameter name of the registry detector
REGISTRY_METHODS = {
    'orb_cv': 'n_features',
    'orb_manual': 'n_features',
    'orb_fast': 'n_features',
    'orb_pyramid': 'n_features',
    'sift_cv': 'n_features',
    'xfeat': 'top_k',
}
ALIKE_METHODS = {'alike_n': 'alike-n', 'alike_l': 'alike-l'}
ALL_METHODS = tuple(REGISTRY_METHODS) + tuple(ALIKE_METHODS)


class ExtractorError(RuntimeError):
    """A method cannot be built (missing dependency or weights)."""


def _add_path(path, *, front=True):
    path = str(Path(path).expanduser().resolve())
    if path in sys.path:
        return
    if front:
        sys.path.insert(0, path)
    else:
        sys.path.append(path)


def _load_registry(xfeat_dir=None):
    # XFeat availability is decided at import time of app.detectors.xfeat
    if xfeat_dir is not None:
        _add_path(xfeat_dir)
    _add_path(DEMO_BACKEND)
    from app import detectors  # noqa: F401
    from app import registry
    return registry


def registry_extractor(registry, method, max_keypoints):
    if method not in registry.DETECTORS:
        raise ExtractorError(f'{method!r} is not registered in the demo registry')
    if not registry.DETECTORS[method]['available']:
        hint = (' (pass --xfeat-dir with a checkout of verlab/accelerated_features'
                ' and install torch)' if method == 'xfeat' else '')
        raise ExtractorError(f'{method!r} is not available{hint}')
    fn = registry.get_detector(method)
    params = {REGISTRY_METHODS[method]: int(max_keypoints)}
    return lambda image_bgr: fn(image_bgr, **params)


def alike_extractor(variant, alike_dir, max_keypoints, device='cpu'):
    """ALIKE-n / ALIKE-l from the FIR branch checkout (`alike_dir`)."""
    if alike_dir is None:
        raise ExtractorError(
            f'{variant!r} needs --alike-dir: a checkout of the FIR branch '
            '(alike.py + models/alike-*.pth)')
    alike_dir = Path(alike_dir).expanduser().resolve()
    if not (alike_dir / 'alike.py').is_file():
        raise ExtractorError(f'alike.py not found in {alike_dir}')
    _add_path(alike_dir, front=False)    # FIR also has metrics.py / evaluate.py
    import alike
    config = alike.configs[ALIKE_METHODS[variant]]
    if not Path(config['model_path']).is_file():
        raise ExtractorError(f'ALIKE weights not found: {config["model_path"]}')
    # scores_th / n_limit as in the FIR evaluate.py; top_k caps the budget
    model = alike.ALike(**config, device=device, top_k=int(max_keypoints),
                        scores_th=0.2, n_limit=5000)

    def extract(image_bgr):
        out = model(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), sub_pixel=True)
        points = np.asarray(out['keypoints'], dtype=np.float32).reshape(-1, 2)
        return points, np.asarray(out['descriptors'], dtype=np.float32), 'l2'

    return extract


def build_extractors(methods, max_keypoints, *, xfeat_dir=None, alike_dir=None):
    """{method: extractor}; every method is built (and so validated) up front."""
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ExtractorError(f'Unknown methods {unknown}. Known: {list(ALL_METHODS)}')
    registry = None
    extractors = {}
    for method in dict.fromkeys(methods):
        if method in ALIKE_METHODS:
            extractors[method] = alike_extractor(method, alike_dir, max_keypoints)
        else:
            registry = registry or _load_registry(xfeat_dir)
            extractors[method] = registry_extractor(registry, method, max_keypoints)
    return extractors
