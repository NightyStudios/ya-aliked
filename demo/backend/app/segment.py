"""Инстанс-сегментация Mask2Former (HuggingFace transformers, CPU).

Модель лениво загружается при первом запросе (и/или при warm-up на старте),
результат — RGBA-оверлей всех инстансов, карта границ областей и список
инстансов с id/bbox.
"""

import base64
import colorsys
import os
import threading
import time
from typing import Any, Dict, List

import cv2
import numpy as np

MODEL_ID = os.getenv("M2F_MODEL_ID", "facebook/mask2former-swin-tiny-coco-instance")
# порог по скору инстанса: 0.4 отсекал часть объектов (на кадре видна была
# лишь одна область), 0.25 оставляет практически все правдоподобные маски
SCORE_THRESHOLD = float(os.getenv("M2F_THRESHOLD", "0.25"))
# маски считаются на уменьшенном кадре (сама модель идёт на 384×384) —
# экономим время пост-обработки и вес ответа; фронт растянуть оверлей на кадр
MAX_SIDE = int(os.getenv("M2F_MAX_SIDE", "800"))
# крошечные инстансы мусорят и оверлей, и карту границ — отбрасываем
MIN_AREA = int(os.getenv("M2F_MIN_AREA", "128"))
MASK_ALPHA = 0.55

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
        processor = Mask2FormerImageProcessor.from_pretrained(MODEL_ID)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
        model.eval()

        id2label = {}
        for k, v in (model.config.id2label or {}).items():
            try:
                id2label[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

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
    из 4 соседей (в т.ч. от фона −1). Считается по карте id отрисованных
    инстансов, поэтому покрывает контуры всех областей, включая внутренние."""
    b = np.zeros(labels.shape, dtype=np.uint8)
    diff_h = labels[1:, :] != labels[:-1, :]
    diff_v = labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= diff_h
    b[:-1, :] |= diff_h
    b[:, 1:] |= diff_v
    b[:, :-1] |= diff_v
    return b * 255


def _render_overlay(seg: np.ndarray, infos: List[dict], id2label: Dict[int, str],
                    min_area: int = MIN_AREA):
    """seg — карта id инстансов; возвращает BGRA-оверлей, инстансы и карту границ."""
    h, w = seg.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    labels = np.full((h, w), -1, dtype=np.int32)   # фон −1, дальше id инстансов
    instances: List[dict] = []

    for info in infos:
        inst_id = int(info.get("id", -1))
        if inst_id < 0:
            continue
        mask = seg == inst_id
        area = int(mask.sum())
        if area < min_area:
            continue

        label_id = int(info.get("label_id", 0))
        b, g, r = _label_color(label_id)
        bgra[mask] = (b, g, r, int(255 * MASK_ALPHA))
        labels[mask] = inst_id

        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(bgra, contours, -1, (255, 255, 255, 255), 1)

        instances.append({
            "id": inst_id,
            "label_id": label_id,
            "label": id2label.get(label_id, f"class_{label_id}"),
            "score": float(info.get("score", 0.0)),
            "area": area,
            "bbox": bbox,
        })

    instances.sort(key=lambda x: x["score"], reverse=True)
    return bgra, instances, _boundary_map(labels)


def segment_image(image_bgr: np.ndarray, threshold: float = SCORE_THRESHOLD) -> Dict[str, Any]:
    """Запускает Mask2Former и возвращает dict для SegmentResponse."""
    st = _load_model()
    torch, model, processor = st["torch"], st["model"], st["processor"]

    small, sw, sh = _resize_for_model(image_bgr)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    inputs = processor(images=rgb, return_tensors="pt")
    with _infer_lock, torch.no_grad():
        outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, target_sizes=[(sh, sw)])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred = results[0] if results else {"segmentation": np.zeros((sh, sw), np.int64),
                                       "segments_info": []}
    seg = pred["segmentation"]
    if hasattr(seg, "cpu"):
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)

    bgra, instances, boundary = _render_overlay(
        seg, pred.get("segments_info") or [], st["id2label"], MIN_AREA)

    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError("Не удалось закодировать оверлей в PNG")
    ok_b, buf_b = cv2.imencode(".png", boundary)
    if not ok_b:
        raise RuntimeError("Не удалось закодировать карту границ в PNG")

    return {
        "overlay_png_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        "boundary_png_b64": base64.b64encode(buf_b.tobytes()).decode("ascii"),
        "instances": instances,
        "width": sw,
        "height": sh,
        "model": MODEL_ID,
        "threshold": threshold,
        "elapsed_ms": dt_ms,
    }
