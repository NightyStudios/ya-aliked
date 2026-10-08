# Восстановление обучения ALIKED

Рабочий код самостоятельной реконструкции по статье, публичной модели ALIKED
и архиву обучения ALIKE. Оригинальный training code ALIKED закрыт.
Обучение и обновление параметров при разработке **не запускались**.
Результаты статьи не считаются воспроизведёнными до обучения и оценки на данных.

Подробное сравнение и источники: [исследование](../../docs/aliked-reproduction/RESEARCH.md).

## Что находится здесь

| Файл | Назначение |
| --- | --- |
| `model.py` | TrainableALIKED: публичный backbone и SDDH, без вызова inference-таймеров CUDA |
| `detector.py` | DKD, top-400 + 400 случайных точек, NMS, differentiable soft argmax |
| `geometry.py` | Homography и SE3/depth, COLMAP `+0.5/-0.5`, видимость и occlusion |
| `losses.py` | Reprojection, dispersity peak, sparse NRE, reliability |
| `data.py` | Существующий MegaDepth DISK и локальные homography/style-transfer пары |
| `validation.py` | Holdout MMA@3 с mutual NN; метрика выбора best checkpoint |
| `cli.py` | Проверка конфигурации/данных, цикл обучения, accumulation, checkpoint/resume |
| `vendor/` | Публичные слои ALIKED с лицензией и SHA256 исходников |

Импорт модулей не запускает обучение. `TrainableALIKED` по умолчанию
инициализируется случайно; `initial_weights` позволяет явно загрузить `.pth`.
Все четыре опубликованные модели поддерживаются, форматы state_dict совместимы.

## Окружение

Из корня проекта:

```bash
uv venv --python 3.12 .venv-aliked
uv pip install --python .venv-aliked/bin/python -r src/aliked_training/requirements.txt
uv pip install --python .venv-aliked/bin/python pytest
```

В текущем workspace окружение `.venv-aliked` уже создано.
PyTorch 2.5.1 + torchvision 0.20.1 проверены на CPU macOS. Историческое
окружение статьи отличается; проверка CUDA этой реконструкции ещё не выполнена.
Сборка `custom_ops` не требуется: извлечение SDDH-патчей использует PyTorch.
Deformable convolution остаётся настоящей `torchvision.ops.deform_conv2d`.

## Без запуска обучения

```bash
PYTHONPATH=src .venv-aliked/bin/python -m aliked_training.cli inspect \
  --config configs/aliked/paper.json
.venv-aliked/bin/python -m pytest tests/aliked_training -q
```

`inspect` печатает итоговую конфигурацию, effective batch size и число
microbatches. Он не создаёт optimizer и не запускает модель.
С заполненными путями `inspect --check-data` дополнительно читает по одной
паре из каждого источника, проверяя изображения, depth и calibration.
Он также требует отдельный validation split.

## Пути для следующего агента / ipynb

Пути намеренно оставлены `null` в JSON: следующий агент адаптирует их
под сервер и notebook. Ничего скачивать не нужно. Загрузчики не содержат
сетевых запросов.

Для MegaDepth DISK задаются:

```json
{
  "megadepth_root": "/server/path/to/megadepth",
  "megadepth_manifest": "dataset.json",
  "validation_megadepth_root": "/server/path/to/imw2020-val"
}
```

Относительные пути внутри `dataset.json` вычисляются от соответствующего root.
Поддерживается schema архива ALIKE:

```json
{
  "0001": {
    "image_path": "scenes/0001/images",
    "depth_path": "scenes/0001/depths",
    "calib_path": "scenes/0001/calibration",
    "images": ["a.jpg", "b.jpg", "c.jpg"],
    "tuples": [[0, 1, 2], [0, 2]]
  }
}
```

Depth: `<stem>.h5`, ключ `depth`. Calibration:
`calibration_<image-name>.h5` (например, `calibration_a.jpg.h5`), либо
`calibration_<stem>.h5`, ключи `K`, `R`, `T`. `R/T` — world-to-camera;
`T01 = pose1 @ inverse(pose0)`. Выбор двух изображений из tuple,
ресайз RGB/depth и изменение intrinsics выполняет загрузчик.
За эпоху выбираются до `pairs_per_scene` разных tuple на сцену;
при меньшем числе tuple искусственного увеличения количества нет.

`homography_manifest` указывает на локальный JSONL. Пути строк считаются
от `homography_root`, а без него — от каталога manifest:

```jsonl
{"image0":"oxford/image.jpg"}
{"image0":"aachen/original.jpg","image1":"aachen/style.jpg","aligned":true}
{"image0":"pairs/a.jpg","image1":"pairs/b.jpg","H01":[[1,0,20],[0,1,5],[0,0,1]]}
```

Одна картинка создаёт синтетическую homography-пару. Для двух картинок
нужен `H01` из исходных pixel coordinates либо `aligned:true` для
геометрически совмещённой style-transfer пары одинакового размера.
Oxford, Paris, Aachen и Aachen style transfer включаются в один manifest.
Известная H преобразуется вместе с ресайзом и дополнительной homography.
Для validation нужны реальные пары с известной H; автоматическая генерация
identity-пар для validation запрещена.

Доля homography-пар определяется `homography_probability` (по умолчанию 0.5).
Это допущение реконструкции, поскольку автор не раскрыл пропорцию источников.
R2D2-style tilt и uniform PixelNoise поддерживаются. Полная цепочка
RandomScale/crop R2D2 не выдаётся за оригинальный dataloader ALIKED:
он не опубликован. Здесь итоговый ресайз 800×800 задан статьёй.

Notebook может использовать модули напрямую:

```python
from aliked_training.config import load_config
from aliked_training.data import DISKMegaDepth, HomographicPairs, MixedPairs
from aliked_training.model import TrainableALIKED
from aliked_training.losses import pair_loss

config = load_config("configs/aliked/paper.json")
# Следующий агент задаёт root/manifest здесь или в JSON.
model = TrainableALIKED(config)
# model(images) -> dict со списками keypoints/descriptors/scores,
# score_dispersity только для surviving DKD-точек и dense score_map.
# pair_loss(pred0, pred1, samples, config) -> rp/pk/ds/re/total/matches.
```

Геометрия хранится по одному dict на sample. `collate_pairs` сохраняет
list samples, что позволяет смешивать `homo` и `se3` в одном batch.
Tensor values переводятся на устройство через `to_device`.

## Настройки и последующий запуск

`paper.json` использует `t_rel=1`; `code-temperature.json` — `t_rel=0.1`
из опубликованного архива ALIKE и обсуждения issue #17. Остальные настройки
разрешаются через `config.DEFAULTS`: 800×800, batch 2, accumulation 6,
100000 optimizer updates, temperatures 0.1/0.1, Adam betas 0.9/0.999,
weights rp/pk/ds/re = 1/0.5/5/1. Learning rate 3e-4 и warmup 500 перенесены
из ALIKE, поскольку статья ALIKED не задаёт их. Интерпретация «100K steps»
как optimizer updates также является допущением: получается 600K microbatches.

`peak_mode="upstream"` сохраняет квадрат расстояния / radius² из public DKD.
`peak_mode="paper"` использует unsquared-distance dot product из Eq. (8).
`reprojection_norm=2` следует Eq. (7); `1` соответствует норме в архиве ALIKE.
`aliked-n16rot` требует явно выбрать `rotation_degrees`: точный диапазон
вращений автора неизвестен; смена только имени модели не включает augmentation.

Код последующего запуска (сейчас **не выполнялся**):

```bash
PYTHONPATH=src python -m aliked_training.cli train --config /path/to/server-config.json
PYTHONPATH=src python -m aliked_training.cli train --config /path/to/server-config.json \
  --resume /path/to/output/last.pt
```

Checkpoint сохраняется после полной accumulation-группы. Он содержит model,
optimizer, scheduler, step/epoch/batch cursor и RNG states; последние
необходимы для восстановления случайных probes. CPU augmentation задаётся
детерминированно на sample/epoch. Для resume загружайте только свой checkpoint.
`max_steps`, `device`, `workers`, `output_dir` можно менять при resume;
остальные настройки проверяются на совпадение.

Выходы: `config.json`, `metrics.jsonl`, `last.pt`, `best.pt`, `best-model.pth`.
Последний формат подходит public inference после загрузки `state_dict`.
Best выбирается по среднему holdout MMA@3; это явно выбранная proxy-метрика,
а не полный IMW benchmark. На следующем этапе нужны проверка на сервере,
обучение, подбор неизвестных настроек и официальный benchmark.
