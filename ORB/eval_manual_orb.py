"""Evaluate manual ORB (slow/fast) and OpenCV baselines on HPatches.

Modes:
  slow       — pure-Python FAST + loop BRIEF (notebook version). Very slow.
  fast       — cv2 FAST + vectorized BRIEF + integral image. Default.
  opencv-orb — OpenCV's cv2.ORB_create.
  opencv-sift— OpenCV's cv2.SIFT_create.

You can combine several modes in one run:
  python eval_manual_orb.py --mode fast opencv-orb opencv-sift ...
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ORB.metrics.local_feature_metrics import evaluate, dataset_sequences, DatasetError
from Красота.backend.ORB.manual_orb import manual_orb_extractor
from Красота.backend.ORB.manual_orb_fast import manual_orb_extractor_fast
from Красота.backend.ORB.manual_orb_pyramid import manual_orb_extractor_pyramid


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def build_manual_slow(args):
    return manual_orb_extractor(
        n_features=args.max_keypoints,
        fast_threshold=args.fast_threshold,
        use_orientation=not args.no_orientation,
        use_cv_fast=False,
    )


def build_manual_fast(args):
    return manual_orb_extractor_fast(
        n_features=args.max_keypoints,
        fast_threshold=args.fast_threshold,
        use_orientation=not args.no_orientation,
        use_cv_fast=not args.manual_fast,
        discretize_angles=not args.no_discretize,
    )

def build_manual_pyramid(args):
    return manual_orb_extractor_pyramid(
        n_features=args.max_keypoints,
        fast_threshold=args.fast_threshold,
        n_levels=args.nlevels,
        scale_factor=args.scale_factor,
        use_orientation=not args.no_orientation,
        use_cv_fast=not args.manual_fast,
        discretize_angles=not args.no_discretize,
    )


def build_opencv_orb(args):
    detector = cv2.ORB_create(
        nfeatures=args.max_keypoints,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=args.fast_threshold,
    )

    def extract(image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        if not kp:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 32), dtype=np.uint8))
        # sort by response, cap at max_keypoints
        order = np.argsort([k.response for k in kp])[::-1][:args.max_keypoints]
        pts = np.asarray([kp[i].pt for i in order], dtype=np.float32)
        des = desc[order] if desc is not None else None
        return pts, des

    return extract


def build_opencv_sift(args):
    detector = cv2.SIFT_create(nfeatures=args.max_keypoints)

    def extract(image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)
        if not kp:
            return (np.zeros((0, 2), dtype=np.float32), None)
        order = np.argsort([k.response for k in kp])[::-1][:args.max_keypoints]
        pts = np.asarray([kp[i].pt for i in order], dtype=np.float32)
        des = desc[order] if desc is not None else None
        return pts, des

    return extract


BUILDERS = {
    'slow':          (build_manual_slow,    True),
    'fast':          (build_manual_fast,    True),
    'fast-pyramid':  (build_manual_pyramid, True),
    'opencv-orb':    (build_opencv_orb,     True),
    'opencv-sift':   (build_opencv_sift,    False),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default="data/hpatches-sequences-release")
    parser.add_argument('--mode', nargs='+',
                        choices=list(BUILDERS.keys()),
                        default=['fast'],
                        help='One or more evaluation backends to run.')
    parser.add_argument('--max-keypoints', type=int, default=1000)
    parser.add_argument('--fast-threshold', type=int, default=20)
    parser.add_argument('--no-orientation', action='store_true',
                        help='Manual modes: disable steered BRIEF.')
    parser.add_argument('--manual-fast', action='store_true',
                        help='fast mode: use pure-Python FAST instead of cv2 FAST.')
    parser.add_argument('--no-discretize', action='store_true',
                        help='fast mode: disable angle discretization cache.')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument('--limit-sequences', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--nlevels', type=int, default=8,
                    help='fast-pyramid: number of pyramid levels.')
    parser.add_argument('--scale-factor', type=float, default=1.2,
                    help='fast-pyramid: scale ratio between levels.')
    args = parser.parse_args()

    try:
        dataset_path, sequences = dataset_sequences(
            args.dataset, limit_sequences=args.limit_sequences)
    except DatasetError as exc:
        parser.error(str(exc))

    print(f'Dataset:   {dataset_path}')
    print(f'Sequences: {len(sequences)}')
    print(f'Modes:     {args.mode}')
    print()

    # Build every extractor before starting the long run.
    extractors = {}
    for mode in dict.fromkeys(args.mode):
        builder, binary = BUILDERS[mode]
        extractors[mode] = (builder(args), binary)

    report = {
        'config': {
            'dataset':       str(dataset_path),
            'sequences':     len(sequences),
            'max_keypoints': args.max_keypoints,
            'fast_threshold': args.fast_threshold,
            'thresholds':    args.thresholds,
            'limit_sequences': args.limit_sequences,
            'orientation':   not args.no_orientation,
        },
        'methods': {},
    }

    for mode, (extract, binary) in extractors.items():
        print(f'=== {mode} ===')
        t0 = time.perf_counter()
        result = evaluate(dataset_path, extract, binary=binary,
                          thresholds=tuple(args.thresholds),
                          limit_sequences=args.limit_sequences)
        elapsed = time.perf_counter() - t0
        print(f'  done in {elapsed:.1f} s '
              f'({elapsed / max(len(sequences) * 6, 1) * 1000:.0f} ms/image)')
        print(json.dumps(result['summary'], indent=2))
        print()
        report['methods'][mode] = result

    out = args.output or Path(__file__).resolve().parent / 'output' / 'orb_eval.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()