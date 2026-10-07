"""Final ALIKE-N/L HPatches evaluation and keypoint export."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm

from alike import ALike, configs
from metrics import metrics

ROOT = Path(__file__).resolve().parent
AUTHOR_EXCLUDED = {
    'i_contruction', 'i_crownnight', 'i_dc', 'i_pencils',
    'i_whitebuilding', 'v_artisans', 'v_astronautis', 'v_talent',
}
FIELDS = ['model', 'seq', 'type', 'img', 'idx', 'x', 'y', 'score']


def select_sequences(dataset):
    if not dataset.is_dir():
        raise ValueError(f'Dataset directory does not exist: {dataset}')
    selected, excluded = [], []
    for sequence in sorted(dataset.iterdir()):
        if not sequence.is_dir() or not sequence.name.startswith(('i_', 'v_')):
            continue
        if sequence.name in AUTHOR_EXCLUDED:
            excluded.append(sequence.name)
            continue
        allowed = True
        for index in range(1, 7):
            image = cv2.imread(str(sequence / f'{index}.ppm'))
            if image is None:
                raise ValueError(f'Missing or unreadable image: {sequence / f"{index}.ppm"}')
            height, width = image.shape[:2]
            if min(height, width) > 1200 or max(height, width) > 1600:
                allowed = False
        for index in range(2, 7):
            if not (sequence / f'H_1_{index}').is_file():
                raise ValueError(f'Missing homography: {sequence / f"H_1_{index}"}')
        if allowed:
            selected.append(sequence)
        else:
            excluded.append(sequence.name)
    if not selected:
        raise ValueError('No complete sequences satisfy the size limit.')
    return selected, excluded


def summarize(rows):
    summary = {'pairs': len(rows)}
    for name in ('mma', 'mha_corner_fraction', 'mha_author_code', 'ms'):
        summary[name] = np.mean([row[name] for row in rows], axis=0).tolist()
    summary['mean_matches'] = float(np.mean([row['matches'] for row in rows]))
    summary['mean_correct_matches'] = np.mean([
        np.array(row['mma']) * row['matches'] for row in rows
    ], axis=0).tolist()
    return summary


@torch.inference_mode()
def run(model_name, sequences, writer):
    model = ALike(**configs[model_name], device='cuda',
                  top_k=0, scores_th=0.2, n_limit=5000)
    cv2.setRNGSeed(0)
    rows, counts = [], []
    for sequence in tqdm(sequences, desc=model_name):
        features, shapes = [], []
        for index in range(1, 7):
            image = cv2.imread(str(sequence / f'{index}.ppm'))
            if image is None:
                raise ValueError(f'Cannot read {sequence / f"{index}.ppm"}')
            shapes.append(image.shape)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = ToTensor()(rgb).unsqueeze(0).cuda()
            dense, score_map = model.extract_dense_map(tensor)
            k, d, scores, _ = model.dkd(score_map, dense, sub_pixel=True)
            points = ((k[0] + 1) / 2 * k[0].new_tensor([
                [rgb.shape[1] - 1, rgb.shape[0] - 1]
            ])).cpu().numpy()
            values = scores[0].cpu().numpy()
            features.append((points, d[0]))
            counts.append(len(points))
            writer.writerows(
                (model_name, sequence.name,
                 'illumination' if sequence.name.startswith('i_') else 'viewpoint',
                 index, idx, format(float(x), '.9g'), format(float(y), '.9g'),
                 format(float(score), '.9g'))
                for idx, ((x, y), score) in enumerate(zip(points, values))
            )
            del dense, score_map, tensor, k, d, scores
        a, da = features[0]
        for target in range(2, 7):
            b, db = features[target - 1]
            if len(da) and len(db):
                similarity = da @ db.T
                ab, ba = similarity.argmax(dim=1), similarity.argmax(dim=0)
                ids = torch.arange(len(da), device='cuda')
                mask = ids == ba[ab]
                matches = torch.stack((ids[mask], ab[mask]), dim=1).cpu().numpy()
                del similarity, ab, ba, ids, mask
            else:
                matches = np.empty((0, 2), dtype=int)
            homography = np.loadtxt(sequence / f'H_1_{target}')
            if homography.shape != (3, 3) or not np.isfinite(homography).all():
                raise ValueError(f'Invalid homography: {sequence / f"H_1_{target}"}')
            rows.append(dict(sequence=sequence.name, target=target,
                             **metrics(a, b, matches, homography,
                                       shapes[0], shapes[target - 1])))
        del features, da, db
        torch.cuda.empty_cache()
    return {
        'model': model_name, 'device': torch.cuda.get_device_name(),
        'scores_th': 0.2, 'top_k': 0, 'n_limit': 5000, 'sub_pixel': True,
        'thresholds_px': [1, 2, 3], 'sequences': len(sequences),
        'images': len(counts), 'keypoints': sum(counts),
        'mean_keypoints': float(np.mean(counts)),
        'summary': summarize(rows), 'pairs': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path,
                        default=ROOT / 'data' / 'hpatches-sequences-release')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--models', nargs='+', choices=['alike-n', 'alike-l'],
                        default=['alike-n', 'alike-l'])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install the CUDA PyTorch build.')
    try:
        sequences, excluded = select_sequences(args.dataset)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'{len(sequences)} sequences, {len(sequences)*6} images, '
          f'{len(sequences)*5} pairs', flush=True)
    with (args.output / 'keypoints.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        for model_name in dict.fromkeys(args.models):
            report = run(model_name, sequences, writer)
            report['excluded_sequences'] = excluded
            report['size_limit'] = {'short_side': 1200, 'long_side': 1600}
            path = args.output / f'{model_name}.json'
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                            encoding='utf-8')
            print(json.dumps(report['summary'], indent=2), flush=True)
            print(f'Saved {path}', flush=True)
    print(f'Saved {args.output / "keypoints.csv"}', flush=True)


if __name__ == '__main__':
    main()
