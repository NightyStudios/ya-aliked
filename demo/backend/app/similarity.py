"""Метрика схожести двух изображений без ключевых точек.

Порт ноутбучной функции similarity(): MAE/PSNR, SSIM (skimage), NCC,
сдвиг всей картинки (фазовая корреляция) и сетка локальных сдвигов
tile×tile. Работает в RGB, как и исходный код.
"""

import math
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity

DEFAULT_TILE = 128


def _finite(x: Any) -> Optional[float]:
    """float или None — JSON не умеет inf/nan."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def compute_metrics(ref: np.ndarray, img: np.ndarray, tile: int = DEFAULT_TILE) -> Dict[str, Any]:
    """ref, img — RGB uint8 одного размера. Возвращает метрики (см. schemas)."""
    if ref.shape != img.shape:
        h = min(ref.shape[0], img.shape[0])
        w = min(ref.shape[1], img.shape[1])
        ref, img = ref[:h, :w], img[:h, :w]
    h, w = ref.shape[:2]
    R, I = ref.astype(np.float32), img.astype(np.float32)
    g1 = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # чёрные поля (нет данных после подгонки) не считаем
    valid = cv2.erode(
        ((ref.max(2) > 5) & (img.max(2) > 5)).astype(np.uint8),
        np.ones((9, 9), np.uint8)) > 0
    if valid.sum() < 100:                       # вырожденный случай — берём всё
        valid = np.ones((h, w), dtype=bool)

    out: Dict[str, Any] = {
        "rmse": None, "mae": None, "psnr": None, "ssim": None, "ncc": None,
        "dx": None, "dy": None, "tiles": 0, "tile_median": None,
        "tiles_under_1px": None,
    }

    # 1) разница яркости по пикселям
    diff = np.abs(R - I).mean(2)
    mae = diff[valid].mean()
    mse = ((R - I) ** 2).mean(2)[valid].mean()
    out["rmse"] = _finite(math.sqrt(mse))
    out["mae"] = _finite(mae)
    out["psnr"] = _finite(10 * np.log10(255 ** 2 / mse)) if mse > 0 else None

    # 2) структурная похожесть (SSIM) и корреляция
    _, ssim_map = structural_similarity(g1, g2, data_range=255, full=True)
    out["ssim"] = _finite(ssim_map[valid].mean())
    a = g1[valid] - g1[valid].mean()
    b = g2[valid] - g2[valid].mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    out["ncc"] = _finite((a * b).sum() / denom) if denom > 0 else None

    # 3) общий сдвиг всей картинки (фазовая корреляция, субпиксельно)
    if h >= 2 and w >= 2:
        f1 = np.where(valid, g1, float(g1[valid].mean()))
        f2 = np.where(valid, g2, float(g2[valid].mean()))
        try:
            (dx, dy), _resp = cv2.phaseCorrelate(f1, f2, cv2.createHanningWindow((w, h), cv2.CV_32F))
            out["dx"], out["dy"] = _finite(dx), _finite(dy)
        except cv2.error:
            pass

    # 4) сдвиг по кускам tile×tile: где картинки разъехались локально
    if tile >= 8 and h >= tile and w >= tile:
        win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
        grid = np.full((h // tile, w // tile), np.nan, np.float32)
        for r in range(h // tile):
            for c in range(w // tile):
                ys, xs = slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile)
                p1, p2 = g1[ys, xs], g2[ys, xs]
                # край или однотонный кусок — сдвиг не определить
                if valid[ys, xs].mean() < 0.95 or p1.std() < 5:
                    continue
                (tx, ty), resp = cv2.phaseCorrelate(p1, p2, win)
                if resp > 0.1:
                    grid[r, c] = np.hypot(tx, ty)
        t = grid[~np.isnan(grid)]
        out["tiles"] = int(t.size)
        if t.size:
            out["tile_median"] = _finite(np.median(t))
            out["tiles_under_1px"] = _finite((t < 1).mean())

    return out


def compare_all(ref: np.ndarray, initial: np.ndarray, warped: np.ndarray,
                tile: int = DEFAULT_TILE) -> Dict[str, Any]:
    """Все три изображения обрезаются до общей области — «начальное» и
    «наложенное» сравниваются с эталоном на одинаковой площади."""
    h = min(ref.shape[0], initial.shape[0], warped.shape[0])
    w = min(ref.shape[1], initial.shape[1], warped.shape[1])
    if h < 8 or w < 8:
        raise ValueError(f"Общая область слишком мала: {w}×{h}")

    t0 = time.perf_counter()
    ref_c, ini_c, wrp_c = (x[:h, :w] for x in (ref, initial, warped))
    initial_m = compute_metrics(ref_c, ini_c, tile)
    warped_m = compute_metrics(ref_c, wrp_c, tile)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ref_size": [int(ref.shape[1]), int(ref.shape[0])],
        "compared_size": [int(w), int(h)],
        "initial": initial_m,
        "warped": warped_m,
        "elapsed_ms": elapsed_ms,
    }


def labels() -> List[str]:
    """Порядок метрик для отображения (документация API/фронтенда)."""
    return ["rmse", "mae", "psnr", "ssim", "ncc", "dx", "dy",
            "tiles", "tile_median", "tiles_under_1px"]
