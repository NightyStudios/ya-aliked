# Обучение ALIKED

Реконструкция по статье ALIKED, публичной модели и архиву обучения ALIKE.
Оригинальный training code ALIKED не опубликован. Все четыре loss сохранены:
`L = rp + 0.5 * pk + 5 * ds + re`.

## Установка и проверка

Команды выполняются **из корня репозитория**:

```bash
python3.12 -m venv .venv-aliked
source .venv-aliked/bin/activate
python -m pip install -r aliked_training/requirements-dev.txt
python -m aliked_training.cli inspect --config configs/aliked/paper.json
python -m pytest tests -q
```

Для GPU установите совместимые CUDA-сборки torch 2.5.1 и torchvision 0.20.1.
Custom C++ patch-extraction extension не нужен; deformable convolution остаётся
`torchvision.ops.deform_conv2d`. Тесты проверены на CPU; работа на CUDA,
полноразмерное обучение и benchmark-качество требуют отдельного запуска.

`inspect` не создаёт модель/optimizer. `inspect --check-data` читает по паре
из каждого источника, проверяет разделение наборов и показывает фактические
размеры, scene IDs, SHA256 manifests и индексы выбранного holdout.

## Конфигурация и данные

Скопируйте `configs/aliked/paper.json` в свой JSON и заполните пути:

```json
{
  "megadepth_root": "/data/megadepth-train",
  "megadepth_manifest": "dataset.json",
  "homography_manifest": "/data/homography/train.jsonl",
  "homography_root": "/data/homography",
  "validation_megadepth_root": "/data/megadepth-val",
  "validation_megadepth_manifest": "dataset.json",
  "validation_homography_manifest": "/data/homography/val.jsonl",
  "validation_homography_root": "/data/homography",
  "output_dir": "output/aliked-experiment"
}
```

Необязательно использовать оба validation-источника, но нужен хотя бы один.
Обучение использует оба источника, с `homography_probability=0.5` по умолчанию.
Настройки, отсутствующие в JSON, берутся из `config.DEFAULTS`.

MegaDepth DISK manifest:

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

Относительные пути разрешаются от соответствующего root. Depth хранится
в `<stem>.h5`, ключ `depth`; calibration — `calibration_<image-name>.h5`
или `calibration_<stem>.h5`, ключи `K`, `R`, `T`. Extrinsics — world-to-camera.
Из каждого tuple выбираются два изображения. `pairs_per_scene` ограничивает
число различных tuples за эпоху, не увеличивает маленькую сцену искусственно.

Depth ресайзится через `nearest-exact`, согласованный с pixel-centre конвенцией
PIL RGB resize и сохраняющий depth holes. Масштабирование K уже согласовано
с COLMAP `+0.5/-0.5`. `depth_tolerance=0.05` — абсолютный допуск в единицах
реконструкции; проверяйте его по доле видимых точек на своих данных.

Homography manifest — JSONL:

```jsonl
{"image0":"oxford/image.jpg"}
{"image0":"aachen/original.jpg","image1":"aachen/style.jpg","aligned":true}
{"image0":"pairs/a.jpg","image1":"pairs/b.jpg","H01":[[1,0,20],[0,1,5],[0,0,1]]}
```

Одна картинка создаёт synthetic pair. Для двух нужны H01 в исходных pixel
coordinates либо aligned=true и одинаковые размеры. Validation требует двух
изображений; синтетические identity-пары из одиночной записи не подставляются.
H01 преобразуется при resize и augmentation. Повторные draws одной записи
в эпохе получают разные, но воспроизводимые augmentation (seed/epoch/draw ID).
Oxford/Paris/Aachen/style-transfer включаются в manifest пользователем.

Проверяются пересечения MegaDepth scene IDs и canonical image paths между
train и validation. Копии и переименованные изображения требуют самостоятельной
проверки идентичности/scene IDs; manifests не заменяют аудит dataset split.

## Валидация и выбор checkpoint

По умолчанию: до **50 пар на сцену**, не более **200 пар на источник**.
Выборка детерминирована `validation_seed`; MegaDepth сцены чередуются,
для homography берётся фиксированная случайная подвыборка. Во время обучения
сохраняется `data-provenance.json` с SHA256 manifests и выбранными индексами.
Validation использует отдельный RNG и не меняет random probes обучения.

Public-compatible detector включён по умолчанию (`inference_detector=upstream`):
оригинальный threshold fallback, temperature .1, один map-NMS. Training DKD
по-прежнему использует top-k, random probes и дополнительный point-NMS.
`inference_detector=training` явно включает прежний путь evaluation; при таком
выборе развёртывайте тот же wrapper, а не только upstream state_dict.

| Метрика | Определение |
| --- | --- |
| `mma3` | Correct MNN / evaluable MNN; обе reprojection errors ≤3px |
| `mean_correct_matches` | Количество correct MNN / число проверенных пар |
| `ms3` | Correct MNN / сумму min(visible0, visible1) по парам |
| `repeatability3` | Геометрические mutual-NN со средней symmetric error <3px / тот же знаменатель |
| `gt_coverage` | Evaluable MNN / все putative MNN |
| `keypoint_gt_coverage` | Точки с известной геометрией / все точки |

Depth holes — unknown и не считаются ошибочными matches. Известная окклюзия
или выход за кадр в любом направлении остаются negative. Если denominator
нулевой, ratio записывается как JSON null. Пустой детектор при доступной GT
получает selection score 0. Источник без evaluable geometry вызывает ошибку,
вместо того чтобы незаметно исчезнуть из среднего по источникам.

По умолчанию best выбирается по **mean_correct_matches**, усреднённому по
источникам с равными весами: один удачный match больше не побеждает сотню
за счёт precision. `validation_selection_metric` также принимает `mma3` и
`ms3` для явных экспериментов. Это matching proxies, **не** полноценные
HPatches homography accuracy или IMW pose mAA. Сравнивайте запуски с одинаковым
holdout, лимитом keypoints и detector; итоговое качество проверяйте downstream.

## Запуск и resume

```bash
python -m aliked_training.cli inspect --config /path/to/server.json --check-data
python -m aliked_training.cli train --config /path/to/server.json
python -m aliked_training.cli train --config /path/to/server.json --resume output/aliked-experiment/last.pt
```

Выходы: `config.json`, `data-provenance.json`, `metrics.jsonl`, `last.pt`,
`best.pt`, `best-model.pth`. Сохранение атомарно, только после полной
accumulation-группы, при validation и в конце. `last.pt` содержит optimizer,
scheduler, RNG, step/epoch/batch cursor. Менять при resume разрешено `max_steps`,
`device`, `workers`, `output_dir`; другие настройки и manifests проверяются.
Hash проверяет manifest, но не каждый байт изображений/depth: не меняйте данные
на месте. Переключение CPU/CUDA не обещает побитовую идентичность.

**Старые full checkpoints несовместимы с новым evaluation/data protocol.**
Для продолжения с прежних весов извлеките model state_dict и используйте
`initial_weights` в новом запуске. История best metric и optimizer не переносятся
молча на изменённый протокол. Загружайте только свои доверенные checkpoints.

## Диагностика и эксперименты

`diagnostics_interval=100` включает диагностику последнего microbatch каждого
сотого update; `0` отключает. Логируются source-specific losses, число matches,
доля пустых пар, visible/unknown ratios, detected/random counts, score stats,
reliability target mean/std, SDDH samples вне изображения. Дополнительные
`score_grad_norm/rp|pk|ds|re` — нормы градиентов отдельных **невзвешенных** losses
на этом microbatch. Они требуют дополнительных autograd-проходов.
Обычные losses усредняются по всем microbatches update. Также пишутся
`microsteps`, `samples_seen` и `update_seconds` без validation/checkpoint I/O.

`paper.json` сохраняет t_rel=1; `code-temperature.json` задаёт .1 как ablation,
а не исправление доказанной ошибки. Архив ALIKE использует другую reliability
формулу (ненормированную exp); одна замена температуры не воспроизводит его.
LR 3e-4 и warmup 500 заимствованы из опубликованного кода ALIKE; в ALIKED
эти настройки не раскрыты. peak_mode=upstream сохраняет public squared
radius-normalized dispersity; paper включает unsquared вариант. L2 default
reprojection можно явно сравнить с L1 архивного ALIKE.

Batch2 × accumulation6 — 12 **пар** на update, но BatchNorm видит отдельно
batch2 изображений каждого направления. 100K optimizer updates означают
600K microbatches и 1.2M пар; это выбранная интерпретация steps статьи.

Point-NMS использует стабильный greedy radius selection с одним переносом
кандидатов на CPU, без NxN матрицы и CUDA scalar-loop. Семантика подавления
сохранена; ускорение нужно измерять на целевом GPU. Public evaluation DKD
сохраняет оригинальный dense unfold. AMP и замена архитектуры в baseline
не добавлены без проверки CUDA/numerical stability.

До полного запуска: визуально проверьте depth warps, overfit 8–16 фиксированных
пар, затем сравните температуры при одинаковом бюджете и нескольких seeds.
План и основания исправлений: [TRAINING_AUDIT.md](TRAINING_AUDIT.md).
