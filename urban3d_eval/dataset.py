"""Urban3D pairs (source.png <-> aug_XX/image.png) with dense GT correspondences.

Layout of one dataset root (an extracted urban3d_part_* archive):

    <root>/tools/geometry.py, raster.cpp   dataset's own decoder
    <root>/tiles/<TILE>/source.png         original RGB
    <root>/tiles/<TILE>/geometry.npz       DSM
    <root>/tiles/<TILE>/aug_00..05/        image.png, metadata.json, masks

The decoder is loaded from the same root, because `metadata.json` pins the
SHA256 of that root's raster.cpp.
"""
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

SOURCE_VISIBLE = 2     # source_status: visible source point
VARIANTS = tuple(f'aug_{i:02d}' for i in range(6))


class DatasetError(ValueError):
    """Dataset is missing, incomplete or has an incompatible layout."""


@dataclass(frozen=True)
class Pair:
    root: Path
    tile: str
    variant: str

    @property
    def city(self):
        return self.tile.split('_')[0]

    @property
    def sample_dir(self):
        return self.root / 'tiles' / self.tile / self.variant

    @property
    def source_path(self):
        return self.root / 'tiles' / self.tile / 'source.png'

    @property
    def target_path(self):
        return self.sample_dir / 'image.png'


@dataclass
class GroundTruth:
    """forward_xy[row, col] = [col, row] of source pixel (col, row) in the target.

    Only pixels with source_status == SOURCE_VISIBLE have a usable correspondence.
    """
    forward_xy: np.ndarray        # (H, W, 2) float32
    source_status: np.ndarray     # (H, W) uint8
    roll_deg: float
    tilt_deg: float


def dataset_roots(paths):
    """Accept dataset roots or folders holding several urban3d_part_* roots."""
    roots = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if (path / 'tiles').is_dir():
            roots.append(path)
            continue
        children = sorted(p for p in path.glob('*') if (p / 'tiles').is_dir()) if path.is_dir() else []
        # an archive extracted into its own folder: <path>/urban3d_part_XX/tiles
        children += sorted(p for p in path.glob('*/*') if (p / 'tiles').is_dir())
        if not children:
            raise DatasetError(f'No Urban3D root (a folder with tiles/) found in {path}')
        roots.extend(children)
    return list(dict.fromkeys(roots))


def dataset_tiles(roots, *, tiles=None, limit_tiles=None):
    """Sorted (root, tile_id) list. A tile id must not repeat across roots."""
    found = {}
    for root in roots:
        for tile_dir in sorted((root / 'tiles').iterdir()):
            if not tile_dir.is_dir():
                continue
            if tile_dir.name in found:
                raise DatasetError(f'Tile {tile_dir.name} is present in {found[tile_dir.name]} and {root}')
            found[tile_dir.name] = root
    selected = sorted(found)
    if tiles is not None:
        unknown = sorted(set(tiles) - set(found))
        if unknown:
            raise DatasetError(f'Unknown tiles: {unknown}')
        selected = [t for t in selected if t in set(tiles)]
    selected = selected[:limit_tiles]
    if not selected:
        raise DatasetError('No tiles left to evaluate.')
    return [(found[t], t) for t in selected]


def tile_pairs(root, tile, variants=VARIANTS):
    pairs = [Pair(root, tile, v) for v in variants]
    for pair in pairs:
        for path in (pair.source_path, pair.target_path, pair.sample_dir / 'metadata.json'):
            if not path.is_file():
                raise DatasetError(f'Urban3D dataset is incomplete: missing {path}')
    return pairs


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f'Cannot read {path}')
    return image


@lru_cache(maxsize=None)
def geometry_module(root):
    """The dataset's own decoder; compiles raster.cpp on first use."""
    path = Path(root) / 'tools' / 'geometry.py'
    if not path.is_file():
        raise DatasetError(f'Decoder not found: {path}')
    spec = importlib.util.spec_from_file_location(f'urban3d_geometry_{abs(hash(str(root)))}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.native()    # compile once in the parent process, workers just load it
    return module


def load_ground_truth(pair):
    """Reconstruct dense maps (checksum-verified by the decoder)."""
    forward_xy, _, _, source_status = geometry_module(pair.root).reconstruct(pair.sample_dir)
    meta = json.loads((pair.sample_dir / 'metadata.json').read_text())['augmentation']
    return GroundTruth(forward_xy=forward_xy, source_status=source_status,
                       roll_deg=float(meta['roll_deg']), tilt_deg=float(meta['tilt_deg']))
