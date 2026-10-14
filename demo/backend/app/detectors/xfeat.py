"""XFeat-детектор (CVPR 2024, verlab/accelerated_features).

Ленивая загрузка модели (singleton, как в segment.py). Дескрипторы —
float32 L2-нормированные, 64-мерные; norm="l2" ловится и JS-матчером
фронта, и бэкендовым LightGlue/bf.
"""

import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..registry import register_detector

_state: Dict[str, Any] = {}
_load_lock = threading.Lock()


def _try_import_xfeat():
    """Импорт XFeat: сначала modules.xfeat с PYTHONPATH, потом torch.hub."""
    try:
        from modules.xfeat import XFeat  # type: ignore
        return XFeat
    except ImportError:
        pass
    try:
        import torch.hub
        return torch.hub.load(
            "verlab/accelerated_features", "XFeat",
            pretrained=True, top_k=4096,
        )
    except Exception:
        return None


def xfeat_available() -> bool:
    """Лёгкая проверка без скачивания: modules.xfeat с PYTHONPATH + torch."""
    try:
        import torch  # noqa: F401
        from modules.xfeat import XFeat  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model(top_k: int, detection_threshold: float):
    key = (top_k, detection_threshold)
    if key in _state:
        return _state[key]
    with _load_lock:
        if key in _state:
            return _state[key]
        XFeat = _try_import_xfeat()
        if XFeat is None:
            raise RuntimeError(
                "XFeat недоступен: установите accelerated_features "
                "(PYTHONPATH) и torch"
            )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = XFeat(top_k=top_k, detection_threshold=detection_threshold)
        model.eval().to(device)
        _state[key] = model
        return model


def _to_tensor(image_bgr: np.ndarray, device):
    import torch
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(image_rgb)
        .permute(2, 0, 1).float().div(255.0)
        .unsqueeze(0).to(device)
    )


def run_xfeat(image_bgr: np.ndarray, *, top_k: int = 4096,
              detection_threshold: float = 0.05, **_):
    """(pts (N,2) float32, desc (N,64) float32, "l2")."""
    import torch
    model = _get_model(top_k, detection_threshold)
    tensor = _to_tensor(image_bgr, model.dev)
    with torch.no_grad():
        out = model.detectAndCompute(tensor, top_k=top_k)[0]
    pts = out["keypoints"].cpu().numpy().astype(np.float32)
    desc = out["descriptors"].cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts, desc / norms, "l2"


register_detector(
    "xfeat", run_xfeat,
    available=xfeat_available(),
    description="XFeat (CVPR 2024) — fast sparse features, 64-d L2",
    group="XFeat",
)
