"""Фильтрация кейпоинтов спутникового снимка по семантическим классам.

Установка: pip install torch torchvision 'transformers>=4.46,<5' pillow numpy scipy

Запуск:
    python run_segmentation.py image.png --points points.json \
        --classes water grass --margin 5 --visualize comparison.png

points.json: [[x, y], ...], координаты в пикселях исходной картинки.
--classes перечисляет УДАЛЯЕМЫЕ классы: 0 other (прочее), 1 water (вода),
2 building (здания), 3 road (дороги), 4 soil (земля), 5 grass (трава).
Можно указать номера, английские или указанные русские названия.
Деревья, сельхозугодья и остальные покрытия относятся к 0.

Точку запрещённого класса оставляем, если до ближайшего разрешённого пикселя
не больше margin пикселей. Расстояние евклидово, до центров пикселей; дробные
координаты сохраняются. Класс точки берётся у ближайшего пикселя (при равенстве
округление вверх). Все не перечисленные классы разрешены, включая 0.
Margin=0 отключает спасение точек. Порядок точек сохраняется.

Выход: kept_points.json (можно изменить через --output).
--visualize необязателен: исходник / наложение / маска; галочки означают
оставленные точки, крестики — удалённые. Другие файлы не создаются.

Python: filter_keypoints(image, points, classes, margin, visualization=None)
возвращает NumPy-массив (M, 2). image — путь либо PIL.Image.

Модель Firework2026/mask2former-satellite при первом запуске автоматически
скачивается с Hugging Face (~432 МБ) и кэшируется. Если рядом со скриптом есть
полный чекпойнт в model/, используется он. Изображение обрабатывается в 384x384,
маска возвращается к исходному размеру. У модели 8 безымянных классов, хотя
README перечисляет 9; применён визуально проверенный порядок OpenEarthMap.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

LABELS = ["other", "water", "building", "road", "soil", "grass"]
RUSSIAN = ["прочее", "вода", "здания", "дороги", "земля", "трава"]
REMAP = np.array([4, 5, 0, 3, 0, 1, 0, 2], dtype=np.uint8)
COLORS = np.array([[0, 0, 0], [42, 151, 242], [239, 80, 106],
                   [255, 215, 65], [153, 101, 58], [155, 214, 65]], dtype=np.uint8)


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
    torch.set_num_threads(4)
    local = Path(__file__).resolve().parent / "model"
    source = str(local) if all((local / f).is_file() for f in
        ["config.json", "preprocessor_config.json", "model.safetensors"]) else "Firework2026/mask2former-satellite"
    processor = Mask2FormerImageProcessor.from_pretrained(source)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(source).eval()
    return torch, processor, model


def segment(image):
    torch, processor, model = load_model()
    with torch.inference_mode():
        outputs = model(**processor(images=image, return_tensors="pt"))
        raw = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)])[0].cpu().numpy()
    return REMAP[raw]


def prepare(points, classes, margin, width, height):
    points = np.asarray(points)
    if points.shape == (0,):
        points = np.empty((0, 2))
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in "iuf":
        raise ValueError("Точки должны быть числовым списком [[x, y], ...]")
    if not np.isfinite(points).all():
        raise ValueError("В точках есть NaN или infinity")
    if np.any((points < 0) | (points >= [width, height])):
        raise ValueError(f"Точки должны находиться внутри картинки {width}x{height}")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Margin должен быть конечным неотрицательным числом")
    names = {name: i for i, name in enumerate(LABELS)}
    names.update({name: i for i, name in enumerate(RUSSIAN)})
    excluded = []
    for value in classes:
        text = str(value).strip().lower()
        class_id = int(text) if text.isdecimal() else names.get(text, -1)
        if class_id not in range(6):
            raise ValueError(f"Неизвестный класс {value!r}. Доступны: {LABELS}")
        excluded.append(class_id)
    return points, excluded


def keep_by_mask(mask, points, excluded, margin):
    """Булева маска оставленных точек; расстояния считаются по исходным (x, y)."""
    h, w = mask.shape
    xy = np.floor(points.astype(float) + 0.5).astype(int)
    xy = np.clip(xy, [0, 0], [w - 1, h - 1])
    keep = ~np.isin(mask[xy[:, 1], xy[:, 0]], excluded)
    if margin > 0 and (~keep).any():
        yy, xx = np.nonzero(~np.isin(mask, excluded))
        if len(xx):
            distance = cKDTree(np.column_stack((xx, yy))).query(points[~keep])[0]
            keep[~keep] = distance <= margin
    return keep


def visualize(image, mask, points, keep, path):
    """Прежние цвета и три панели, с галочками и крестиками поверх предсказаний."""
    original = np.asarray(image)
    colored = COLORS[mask]
    overlay = np.rint(original * 0.52 + colored * 0.48).astype(np.uint8)
    overlay[mask == 0] = original[mask == 0]
    panel = Image.new("RGB", (1536, 660), "#141b22")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=18)
    for col, (array, title) in enumerate(zip([original, overlay, colored],
        ["Original", "Overlay + keypoints", "Class mask + keypoints"])):
        tile = Image.fromarray(array).resize((512, 512),
            Image.Resampling.NEAREST if col == 2 else Image.Resampling.BILINEAR)
        marks = ImageDraw.Draw(tile)
        if col:
            for (x, y), kept in zip(points, keep):
                x, y = x * 512 / image.width, y * 512 / image.height
                lines = [[(x-4, y), (x-1, y+4), (x+5, y-4)]] if kept else [
                    [(x-4, y-4), (x+4, y+4)], [(x-4, y+4), (x+4, y-4)]]
                for line in lines:
                    marks.line(line, fill="black", width=5)
                    marks.line(line, fill="white" if kept else "#ff5b36", width=2)
        panel.paste(tile, (col * 512, 32))
        draw.text((col * 512 + 12, 8), title, font=font, fill="white")
    for i, name in enumerate(LABELS):
        x, y = (i % 3) * 512 + 12, 558 + (i // 3) * 28
        draw.rectangle((x, y, x+18, y+18), fill=tuple(COLORS[i]), outline="#666666")
        draw.text((x+26, y), f"{i}: {name}", font=font, fill="white")
    draw.text((12, 620), f"Check = kept ({keep.sum()}); cross = rejected ({(~keep).sum()})", font=font, fill="white")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def filter_keypoints(image, points, classes, margin=0, visualization=None):
    """Вернуть оставшиеся (x, y); classes — исключения, margin — пиксели исходника."""
    if not isinstance(image, Image.Image):
        with Image.open(image) as source:
            image = source.convert("RGB")
    else:
        image = image.convert("RGB")
    points, excluded = prepare(points, classes, margin, image.width, image.height)
    mask = segment(image)
    keep = keep_by_mask(mask, points, excluded, margin)
    if visualization is not None:
        visualize(image, mask, points, keep, visualization)
    return points[keep].copy()


def main():
    parser = argparse.ArgumentParser(description="Фильтр кейпоинтов по семантическим классам")
    parser.add_argument("image")
    parser.add_argument("--points", required=True, help="JSON со списком [[x,y],...]")
    parser.add_argument("--classes", nargs="*", default=[], help="Классы для удаления, названия или ID 0–5")
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--output", default="kept_points.json")
    parser.add_argument("--visualize", nargs="?", const="comparison.png", metavar="PNG")
    args = parser.parse_args()
    try:
        points = json.loads(Path(args.points).read_text())
        kept = filter_keypoints(args.image, points, args.classes, args.margin, args.visualize)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(kept.tolist(), indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Оставлено {len(kept)} из {len(points)}. Результат: {args.output}")


if __name__ == "__main__":
    main()
