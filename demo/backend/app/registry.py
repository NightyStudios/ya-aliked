from typing import Any, Callable, Dict, Tuple

import numpy as np

# Detector signature:
#   fn(image_bgr: np.ndarray, **params) -> (points (N,2) float32,
#                                            desc   (N,D) uint8|float32,
#                                            norm   "hamming"|"l2")
DetectorFn = Callable[..., Tuple[np.ndarray, np.ndarray, str]]

# Matcher signature:
#   fn(desc_a, desc_b, norm, **matcher_params) -> list[(a_idx, b_idx, dist)]
MatcherFn = Callable[..., list]


DETECTORS: Dict[str, Dict[str, Any]] = {}
MATCHERS: Dict[str, MatcherFn] = {}


def register_detector(name, fn, *, available=True, description="",
                      hidden=False, group="ORB"):
    DETECTORS[name] = {
        "fn": fn, "available": available,
        "description": description,
        "hidden": hidden, "group": group,
    }


def register_matcher(name: str, fn: MatcherFn) -> None:
    MATCHERS[name] = fn


def get_detector(name: str) -> DetectorFn:
    if name not in DETECTORS:
        raise KeyError(f"Unknown detector: {name!r}. Known: {list(DETECTORS)}")
    meta = DETECTORS[name]
    if not meta["available"]:
        raise NotImplementedError(f"Detector {name!r} is registered but not yet implemented")
    return meta["fn"]


def get_matcher(name: str) -> MatcherFn:
    if name not in MATCHERS:
        raise KeyError(f"Unknown matcher: {name!r}. Known: {list(MATCHERS)}")
    return MATCHERS[name]