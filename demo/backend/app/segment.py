"""Семантическая сегментация спутниковых снимков Mask2Former (HuggingFace transformers, CPU).

Модель лениво загружается при первом запросе (и/или при warm-up на старте),
Каждый пиксель получает класс. Результат — RGBA-оверлей, карта границ
классов и список связных областей с id/bbox.
"""

import base64
import colorsys
import os
import threading
import time
from typing import Any, Dict, List

import cv2
import numpy as np

MODEL_ID = os.getenv("M2F_MODEL_ID", "Firework2026/mask2former-satellite")
MODEL_SOURCE = os.getenv("M2F_MODEL_PATH", MODEL_ID)
# Размер выходной маски; вход модели определяется её процессором.
MAX_SIDE = int(os.getenv("M2F_MAX_SIDE", "800"))
MASK_ALPHA = 0.55
# Zero-based OpenEarthMap order, also used in output/satellite-segmentation.
# This checkpoint's config has eight LABEL_n entries; its card lists nine.
SATELLITE_LABELS = ["Голая земля", "Трава", "Площадки и покрытия", "Дороги",
                    "Деревья", "Вода", "Поля", "Здания"]

_state: Dict[str, Any] = {}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _load_model() -> Dict[str, Any]:
    """Ленивая загрузка модели и процессора; повторные вызовы отдают кэш."""
    if _state:
        return _state
    with _load_lock:
        if _state:
            return _state
        import torch
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        t0 = time.perf_counter()
        processor = Mask2FormerImageProcessor.from_pretrained(MODEL_SOURCE)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_SOURCE)
        model.eval()

        id2label = {}
        for k, v in (model.config.id2label or {}).items():
            try:
                id2label[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        if MODEL_ID == "Firework2026/mask2former-satellite" and len(id2label) == 8:
            id2label = dict(enumerate(SATELLITE_LABELS))

        _state.update(processor=processor, model=model, torch=torch, id2label=id2label)
        print(f"[segment] {MODEL_ID} загружена за {time.perf_counter() - t0:.1f}s", flush=True)
    return _state


def warmup() -> None:
    """Фоновая подгрузка модели на старте — чтобы первый запрос не ждал скачивание."""
    try:
        _load_model()
    except Exception as e:  # noqa: BLE001 — warm-up не должен ронять сервер
        print(f"[segment] warm-up не удался: {e}", flush=True)


def _label_color(label_id: int) -> tuple:
    """Детерминированный цвет класса (golden ratio по оттенку). BGR для cv2."""
    hue = (label_id * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _resize_for_model(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale >= 1.0:
        return image_bgr, w, h
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA), nw, nh


def _boundary_map(labels: np.ndarray) -> np.ndarray:
    """Карта границ областей: белый (255), если пиксель отличается от любого
    из 4 соседей. Считается по полной карте классов, включая мелкие области."""
    b = np.zeros(labels.shape, dtype=np.uint8)
    diff_h = labels[1:, :] != labels[:-1, :]
    diff_v = labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= diff_h
    b[:-1, :] |= diff_h
    b[:, 1:] |= diff_v
    b[:, :-1] |= diff_v
    return b * 255


def _render_overlay(seg: np.ndarray, id2label: Dict[int, str]):
    """Полная карта классов → BGRA, связные области и границы классов.

    Класс 0 и маленькие области тоже отрисовываются, без прозрачных дыр.
    Названия берём из config модели, не угадываем их по номеру класса.
    Поле instances сохраняет совместимость API и содержит связные области.
    """
    h, w = seg.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    instances: List[dict] = []
    for label_id in np.unique(seg):
        label_id = int(label_id)
        mask = seg == label_id
        b, g, r = _label_color(label_id)
        bgra[mask] = (b, g, r, int(255 * MASK_ALPHA))
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=4)
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            instances.append({
                "id": len(instances),
                "label_id": label_id,
                "label": id2label.get(label_id, f"class_{label_id}"),
                "score": None,
                "area": area,
                "bbox": [x, y, x + width, y + height],
            })
    boundary = _boundary_map(seg)
    bgra[boundary > 0] = (255, 255, 255, 255)
    return bgra, instances, boundary


def segment_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Запускает Mask2Former и возвращает dict для SegmentResponse."""
    t0 = time.perf_counter()
    st = _load_model()
    torch, model, processor = st["torch"], st["model"], st["processor"]

    small, sw, sh = _resize_for_model(image_bgr)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt")
    with _infer_lock, torch.inference_mode():
        outputs = model(**inputs)
        full_seg = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_bgr.shape[:2]])[0].cpu().numpy()
        seg = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(sh, sw)])[0]
    if hasattr(seg, "cpu"):
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)

    bgra, instances, boundary = _render_overlay(seg, st["id2label"])

    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError("Не удалось закодировать оверлей в PNG")
    ok_b, buf_b = cv2.imencode(".png", boundary)
    if not ok_b:
        raise RuntimeError("Не удалось закодировать карту границ в PNG")
    if full_seg.min() < 0 or full_seg.max() > 255:
        raise RuntimeError("Карта классов должна содержать ID от 0 до 255")
    ok_c, buf_c = cv2.imencode(".png", full_seg.astype(np.uint8))
    if not ok_c:
        raise RuntimeError("Не удалось закодировать карту классов")
    class_ids, areas = np.unique(full_seg, return_counts=True)
    present = dict(zip(map(int, class_ids), map(int, areas)))
    classes = [
        {"id": label_id, "label": label,
         "color": "#%02x%02x%02x" % _label_color(label_id)[::-1],
         "area": present.get(label_id, 0)}
        for label_id, label in sorted(st["id2label"].items())
    ]

    return {
        "overlay_png_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        "boundary_png_b64": base64.b64encode(buf_b.tobytes()).decode("ascii"),
        "class_map_png_b64": base64.b64encode(buf_c.tobytes()).decode("ascii"),
        "classes": classes,
        "instances": instances,
        "width": sw,
        "height": sh,
        "model": MODEL_ID,
        "mode": "semantic",
        "threshold": None,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }
