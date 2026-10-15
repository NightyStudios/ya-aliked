"""Evaluate keypoint methods on Urban3D: MMA / MHA over source <-> aug pairs.

    python -m urban3d_eval.evaluate --data <folder with urban3d_part_* roots> \\
        --xfeat-dir <accelerated_features checkout> --alike-dir <FIR checkout>

Run from the repository root. Rows are appended to <output>/per_pair.jsonl as
they are computed, so an interrupted run resumes where it stopped. The tables
are built from that file by `python -m urban3d_eval.report`.

Matching is the same for every method: mutual nearest neighbours (BFMatcher
crossCheck; Hamming for binary, cosine for float descriptors), exactly as in
ORB/metrics/local_feature_metrics.py.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ORB.metrics.local_feature_metrics import match
from urban3d_eval.dataset import (
    DatasetError, dataset_roots, dataset_tiles, geometry_module,
    load_ground_truth, load_image, tile_pairs,
)
from urban3d_eval.extractors import ALL_METHODS, ExtractorError, build_extractors
from urban3d_eval.metrics import pair_metrics, reference_homography

DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'results'


def _ground_truth(pair):
    """Worker: dense GT + reference homography of one pair."""
    gt = load_ground_truth(pair)
    reference, inlier_ratio = reference_homography(gt)
    return gt, reference, inlier_ratio


def _extract(extractor, image):
    start = time.perf_counter()
    points, descriptors, norm = extractor(image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError('Extractor returned non-finite keypoints.')
    if descriptors is not None and len(descriptors) != len(points):
        raise ValueError('Extractor must return one descriptor per keypoint.')
    return points, descriptors, norm, elapsed_ms


def _read_rows(path):
    if not path.is_file():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _config(args, roots, tile_ids):
    return {
        'roots': [r.name for r in roots], 'tiles': len(tile_ids),
        'methods': list(dict.fromkeys(args.methods)),
        'max_keypoints': args.max_keypoints, 'thresholds_px': list(args.thresholds),
        'opencv': cv2.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', nargs='+', type=Path, required=True,
                        help='Urban3D roots (folders with tiles/) or folders containing them.')
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument('--max-keypoints', type=int, default=2000,
                        help='Keypoint budget per image, same for every method.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[1, 2, 3, 4, 5])
    parser.add_argument('--xfeat-dir', type=Path, default=None,
                        help='Checkout of verlab/accelerated_features (for xfeat).')
    parser.add_argument('--alike-dir', type=Path, default=None,
                        help='Checkout of the FIR branch (for alike_n / alike_l).')
    parser.add_argument('--tiles', nargs='+', default=None)
    parser.add_argument('--limit-tiles', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4,
                        help='Processes decoding the ground truth.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.max_keypoints <= 0 or not args.thresholds or any(t <= 0 for t in args.thresholds):
        parser.error('--max-keypoints and --thresholds must be positive.')

    try:
        roots = dataset_roots(args.data)
        tiles = dataset_tiles(roots, tiles=args.tiles, limit_tiles=args.limit_tiles)
        extractors = build_extractors(args.methods, args.max_keypoints,
                                      xfeat_dir=args.xfeat_dir, alike_dir=args.alike_dir)
        for root in roots:
            geometry_module(root)     # compile the rasteriser before workers start
    except (DatasetError, ExtractorError) as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / 'per_pair.jsonl'
    config_path = args.output / 'run_config.json'
    config = _config(args, roots, [t for _, t in tiles])
    if config_path.is_file() and rows_path.is_file():
        previous = json.loads(config_path.read_text())
        for key in ('max_keypoints', 'thresholds_px'):
            if previous.get(key) != config[key]:
                parser.error(f'{rows_path} was produced with {key}={previous.get(key)}; '
                             'use another --output or delete it.')
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    done = {(r['method'], r['tile'], r['variant']) for r in _read_rows(rows_path)}
    print(f'Roots:   {len(roots)}  Tiles: {len(tiles)}  Methods: {list(extractors)}')
    print(f'Already computed: {len(done)} rows\n')

    cv2.setRNGSeed(0)
    thresholds = tuple(args.thresholds)
    started = time.perf_counter()

    def pending(tile_pairs_):
        return [p for p in tile_pairs_ if any((m, p.tile, p.variant) not in done for m in extractors)]

    all_pairs = {tile: tile_pairs(root, tile) for root, tile in tiles}
    with ProcessPoolExecutor(max_workers=args.workers) as pool, rows_path.open('a') as sink:
        futures = {}

        def submit(tile):
            for pair in pending(all_pairs[tile]):
                futures[(pair.tile, pair.variant)] = pool.submit(_ground_truth, pair)

        tile_ids = [t for _, t in tiles]
        submit(tile_ids[0])
        for index, tile in enumerate(tile_ids):
            if index + 1 < len(tile_ids):
                submit(tile_ids[index + 1])      # decode the next tile while this one runs
            todo = pending(all_pairs[tile])
            if not todo:
                continue
            source = load_image(all_pairs[tile][0].source_path)
            truths = {p.variant: futures.pop((p.tile, p.variant)).result() for p in todo}

            for method, extractor in extractors.items():
                if all((method, p.tile, p.variant) in done for p in todo):
                    continue
                points_a, desc_a, norm, ms_a = _extract(extractor, source)
                binary = norm == 'hamming'
                for pair in todo:
                    if (method, pair.tile, pair.variant) in done:
                        continue
                    gt, reference, inlier_ratio = truths[pair.variant]
                    target = load_image(pair.target_path)
                    points_b, desc_b, _, ms_b = _extract(extractor, target)
                    matches = match(desc_a, desc_b, binary)
                    row = pair_metrics(points_a, points_b, matches, gt, reference,
                                       source.shape, thresholds)
                    row.update(
                        method=method, tile=pair.tile, city=pair.city, variant=pair.variant,
                        roll_deg=gt.roll_deg, tilt_deg=gt.tilt_deg,
                        keypoints_a=len(points_a), keypoints_b=len(points_b),
                        extract_ms_a=ms_a, extract_ms_b=ms_b,
                        reference_inlier_ratio=inlier_ratio,
                    )
                    sink.write(json.dumps(row, allow_nan=False) + '\n')
                    sink.flush()
                    done.add((method, pair.tile, pair.variant))
            print(f'[{index + 1}/{len(tile_ids)}] {tile}  '
                  f'{time.perf_counter() - started:.0f} s elapsed', flush=True)

    print(f'\nRows: {rows_path}\nNext: python -m urban3d_eval.report --output {args.output}')


if __name__ == '__main__':
    sys.exit(main())
