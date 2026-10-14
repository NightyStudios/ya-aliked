"""LightGlue-матчер для L2-дескрипторов (xfeat и др. 64-d float).

Использует LighterGlue из accelerated_features (конфиг/веса под XFeat,
kornia — только как бэкенд attention-слоёв). Работает только через
/api/pipeline: нужны координаты точек и размеры изображений (W, H),
которые MatcherFn получает из контекста.
"""

import numpy as np

from ..registry import register_matcher

_LIGHTGLUE = None
_LIGHTGLUE_LOCK = None


def lightglue_available() -> bool:
    try:
        import torch  # noqa: F401
        import kornia  # noqa: F401
        from modules.lighterglue import LighterGlue  # noqa: F401
        return True
    except ImportError:
        return False


def _get_matcher():
    global _LIGHTGLUE, _LIGHTGLUE_LOCK
    if _LIGHTGLUE is not None:
        return _LIGHTGLUE
    if _LIGHTGLUE_LOCK is None:
        import threading
        _LIGHTGLUE_LOCK = threading.Lock()
    with _LIGHTGLUE_LOCK:
        if _LIGHTGLUE is not None:
            return _LIGHTGLUE
        from modules.lighterglue import LighterGlue
        _LIGHTGLUE = LighterGlue()
        return _LIGHTGLUE


def lightglue_match(desc_a, desc_b, norm,
                    pts_a=None, pts_b=None,
                    image_size_a=None, image_size_b=None,
                    min_conf: float = 0.1, **_):
    if norm != "l2":
        raise ValueError(
            f"lightglue требует norm='l2', получено {norm!r} "
            "(матчер рассчитан на xfeat-дескрипторы)"
        )
    if pts_a is None or pts_b is None or image_size_a is None or image_size_b is None:
        raise ValueError(
            "lightglue требует pts_* и image_size_* (вызывается через /api/pipeline)"
        )
    if len(desc_a) == 0 or len(desc_b) == 0:
        return []

    import torch
    lg = _get_matcher()
    dev = lg.dev

    da = torch.from_numpy(np.asarray(desc_a, dtype=np.float32)).to(dev)
    db = torch.from_numpy(np.asarray(desc_b, dtype=np.float32)).to(dev)
    pa = torch.from_numpy(np.asarray(pts_a, dtype=np.float32)).to(dev)
    pb = torch.from_numpy(np.asarray(pts_b, dtype=np.float32)).to(dev)
    sa = torch.tensor(image_size_a, dtype=torch.float32, device=dev)
    sb = torch.tensor(image_size_b, dtype=torch.float32, device=dev)

    with torch.inference_mode():
        out = lg({
            "keypoints0": pa[None], "keypoints1": pb[None],
            "descriptors0": da[None], "descriptors1": db[None],
            "image_size0": sa[None], "image_size1": sb[None],
        }, min_conf=min_conf)

    idxs = out["matches"][0]      # [Si x 2] пары индексов
    scores = out["scores"][0]     # [Si] confidence

    return [
        (int(ia), int(ib), float(1.0 - s))
        for (ia, ib), s in zip(idxs.cpu().numpy(), scores.cpu().numpy())
    ]


register_matcher("lightglue", lightglue_match)
