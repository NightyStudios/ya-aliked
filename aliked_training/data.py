"""Local data only: DISK dataset.json and R2D2-style homographic pairs."""
import json
import random
import functools
import bisect
import itertools
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, ToTensor


def sample_seed(dataset, index, draw_id=None):
    if draw_id is None:
        return dataset.seed + dataset.epoch * len(dataset) + index
    # Include the draw as well as the record: sampling a small manifest with
    # replacement must not replay identical augmentations throughout an epoch.
    key = f"{dataset.seed}:{dataset.epoch}:{index}:{draw_id}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2**63)


def seeded_sample(method):
    """Stateless CPU augmentation allows repeatable worker and resume behaviour."""
    @functools.wraps(method)
    def wrapped(self, index, *, draw_id=None):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(sample_seed(self, index, draw_id))
            if draw_id is None:
                return method(self, index)
            return method(self, index, draw_id=draw_id)
    return wrapped


def image_tensor(path, size, augment):
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        if augment:
            image = ColorJitter(0.2, 0.2, 0.2, 0.1)(image)
        return ToTensor()(image), (height, width)


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


class DISKMegaDepth(Dataset):
    """Read the format used by ALIKE's released MegaDepth loader, without caches."""
    def __init__(self, root, size=800, pairs_per_scene=10000, augment=True,
                 seed=0, excluded_scenes=(), manifest="dataset.json", depth_tolerance=0.05):
        self.root, self.size, self.seed = Path(root), size, seed
        self.augment, self.epoch = augment, 0
        self.pairs_per_scene = pairs_per_scene
        if not np.isfinite(depth_tolerance) or depth_tolerance <= 0:
            raise ValueError("depth_tolerance must be finite and positive")
        self.depth_tolerance = depth_tolerance
        with open(resolve(self.root, manifest)) as stream:
            scenes = json.load(stream)
        self.scenes = [(name, scene) for name, scene in scenes.items() if name not in excluded_scenes]
        if not self.scenes:
            raise ValueError("DISK manifest has no eligible scenes")
        for name, scene in self.scenes:
            for field in ("images", "tuples", "image_path", "depth_path", "calib_path"):
                if field not in scene:
                    raise ValueError(f"DISK scene {name}: missing {field}")
            if not scene["tuples"] or any(len(t) < 2 for t in scene["tuples"]):
                raise ValueError(f"DISK scene {name}: tuples must contain at least two image indices")
        self.counts = [min(pairs_per_scene, len(scene["tuples"])) for _, scene in self.scenes]
        self.ends = list(itertools.accumulate(self.counts))
        self._orders = {}

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._orders = {}

    def __len__(self):
        return self.ends[-1]

    def _read(self, scene, index):
        name = scene["images"][index]
        if not Path(name).suffix:
            name += ".jpg"
        stem = Path(name).stem
        image, original_shape = image_tensor(resolve(self.root, scene["image_path"])/name,
                                             self.size, self.augment)
        depth_path = resolve(self.root, scene["depth_path"])/(stem + ".h5")
        with h5py.File(depth_path, "r") as file:
            depth = torch.from_numpy(np.asarray(file["depth"], dtype=np.float32))
        if tuple(depth.shape) != original_shape:
            raise ValueError(f"Image/depth size mismatch: {depth_path}")
        calibration_dir = resolve(self.root, scene["calib_path"])
        calibration = calibration_dir / ("calibration_" + name + ".h5")
        if not calibration.exists():
            calibration = calibration_dir / ("calibration_" + stem + ".h5")
        with h5py.File(calibration, "r") as file:
            intrinsic = torch.from_numpy(np.asarray(file["K"], dtype=np.float32))
            pose = torch.eye(4)
            pose[:3, :3] = torch.from_numpy(np.asarray(file["R"], dtype=np.float32))
            pose[:3, 3] = torch.from_numpy(np.asarray(file["T"], dtype=np.float32).reshape(3))
        if not torch.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f"Invalid depth values: {depth_path}")
        height, width = original_shape
        intrinsic[0] *= self.size / width
        intrinsic[1] *= self.size / height
        # Match PIL's pixel-centre resize convention while preserving depth holes.
        depth = F.interpolate(depth[None, None], (self.size, self.size), mode="nearest-exact")[0, 0]
        return image, depth, intrinsic, pose

    @seeded_sample
    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        scene_index = bisect.bisect_right(self.ends, index)
        scene_name, scene = self.scenes[scene_index]
        local_index = index - (self.ends[scene_index-1] if scene_index else 0)
        if scene_index not in self._orders:
            order_rng = random.Random(self.seed + self.epoch * len(self.scenes) + scene_index)
            self._orders[scene_index] = order_rng.sample(range(len(scene["tuples"])), self.counts[scene_index])
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        # DISK tuples can contain triplets. Match ALIKE by selecting two members.
        a, b = rng.sample(scene["tuples"][self._orders[scene_index][local_index]], 2)
        image0, depth0, k0, pose0 = self._read(scene, a)
        image1, depth1, k1, pose1 = self._read(scene, b)
        t01 = pose1 @ torch.linalg.inv(pose0)  # world-to-camera matrices
        common = {"mode": "se3", "depth0": depth0, "depth1": depth1,
                  "K0": k0, "K1": k1, "T01": t01,
                  "depth_tolerance": self.depth_tolerance}
        reverse = {"mode": "se3", "depth0": depth1, "depth1": depth0,
                   "K0": k1, "K1": k0, "T01": torch.linalg.inv(t01),
                   "depth_tolerance": self.depth_tolerance}
        return {"image0": image0, "image1": image1, "warp01": common,
                "warp10": reverse, "source": "megadepth", "scene": scene_name}


def homography_from_corners(source, target):
    rows, rhs = [], []
    for (x, y), (u, v) in zip(source, target):
        rows.extend(([x, y, 1, 0, 0, 0, -u*x, -u*y],
                     [0, 0, 0, x, y, 1, -v*x, -v*y]))
        rhs.extend((u, v))
    coefficients = torch.linalg.solve(torch.tensor(rows, dtype=torch.float64),
                                      torch.tensor(rhs, dtype=torch.float64))
    return torch.cat((coefficients, coefficients.new_ones(1))).reshape(3, 3).float()


def warp_image(image, homography):
    """H maps source pixels to destination pixels; grid_sample needs its inverse."""
    _, height, width = image.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    points = torch.stack((x, y, torch.ones_like(x)), -1).to(image)
    source = points @ torch.linalg.inv(homography).T
    grid = source[..., :2] / source[..., 2:]
    grid = grid / image.new_tensor([width-1, height-1]) * 2 - 1
    return F.grid_sample(image[None], grid[None], align_corners=True)[0]


def random_tilting(size, magnitude, rng):
    """R2D2's four-direction tilt distribution, expressed as a forward H."""
    if not magnitude:
        return torch.eye(3)
    corners = torch.tensor([[0., 0.], [float(size), 0.],
                            [float(size), float(size)], [0., float(size)]])
    target = corners.clone()
    amount = rng.randint(1, int(np.ceil(size * magnitude)))
    direction = rng.randrange(4)
    if direction < 2:
        edge = [0, 3] if direction == 0 else [1, 2]
        target[edge, 1] += torch.tensor([-amount, amount])
    else:
        edge = [0, 1] if direction == 2 else [3, 2]
        target[edge, 0] += torch.tensor([-amount, amount])
    return homography_from_corners(corners.tolist(), target.tolist())


class HomographicPairs(Dataset):
    """Local JSONL: {image0, image1?, H01?}, including aligned style transfers.

    If image1 is absent, make a synthetic pair from image0. H01 defaults to
    identity only for an explicitly aligned original/style image pair. See README.
    """
    def __init__(self, manifest, root=None, size=800, augment=True, seed=0,
                 tilt=0.5, rotation=0.0):
        self.manifest = Path(manifest)
        self.root = Path(root) if root else self.manifest.parent
        self.size, self.augment, self.seed, self.epoch = size, augment, seed, 0
        self.tilt, self.rotation = tilt, rotation
        with open(manifest) as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records or any("image0" not in row for row in self.records):
            raise ValueError("Homography manifest must contain image0 in each row")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    @seeded_sample
    def __getitem__(self, index, *, draw_id=None):
        row = self.records[index]
        rng = random.Random(sample_seed(self, index, draw_id))
        image0, shape0 = image_tensor(resolve(self.root, row["image0"]), self.size, self.augment)
        image1, shape1 = image_tensor(resolve(self.root, row.get("image1", row["image0"])),
                                    self.size, self.augment)
        if "image1" in row and "H01" not in row and not row.get("aligned", False):
            raise ValueError("Paired images require H01 or aligned=true (for style transfer)")
        if row.get("aligned", False) and shape0 != shape1 and "H01" not in row:
            raise ValueError("Aligned style transfers must have equal dimensions or an explicit H01")
        h = torch.tensor(row.get("H01", np.eye(3).tolist()), dtype=torch.float32)
        if h.shape != (3, 3) or not torch.isfinite(h).all() or torch.linalg.det(h).abs() < 1e-8:
            raise ValueError("H01 must be a finite, nonsingular 3x3 matrix")
        # PIL resize uses pixel centres: x'=(x+.5)*scale-.5.
        def resize_matrix(shape):
            height, width = shape
            sx, sy = self.size / width, self.size / height
            return torch.tensor([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
        h = resize_matrix(shape1) @ h @ torch.linalg.inv(resize_matrix(shape0))
        if self.augment:
            s = self.size - 1
            transform = random_tilting(self.size, self.tilt, rng)
            if self.rotation:
                angle = np.deg2rad(rng.uniform(-self.rotation, self.rotation))
                c, sine = float(np.cos(angle)), float(np.sin(angle))
                rotation = torch.tensor([[c, -sine, (1-c+sine)*s/2],
                                         [sine, c, (1-c-sine)*s/2], [0., 0., 1.]])
                transform = rotation @ transform
            image1 = warp_image(image1, transform)
            h = transform @ h
            # R2D2 PixelNoise(25) uses uniform noise, followed by uint8 conversion.
            noise = torch.rand_like(image1) * 25 - 12
            image1 = ((image1*255 + noise).clamp(0, 255).floor()) / 255
        shape = (self.size, self.size)
        return {"image0": image0, "image1": image1,
                "warp01": {"mode": "homo", "H": h, "shape1": shape},
                "warp10": {"mode": "homo", "H": torch.linalg.inv(h), "shape1": shape},
                "source": "homography"}


class MixedPairs(Dataset):
    """Configurable source ratio; epoch size is supplied by the perspective set."""
    def __init__(self, perspective, homographic, homography_probability=0.5, seed=0):
        self.perspective, self.homographic = perspective, homographic
        self.probability, self.seed, self.epoch = homography_probability, seed, 0
        if not 0 < self.probability < 1:
            raise ValueError("Use a probability strictly between 0 and 1 for mixed data")

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.perspective.set_epoch(epoch)
        self.homographic.set_epoch(epoch)

    def __len__(self):
        return len(self.perspective)

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        if rng.random() < self.probability:
            return self.homographic.__getitem__(rng.randrange(len(self.homographic)), draw_id=index)
        return self.perspective[index]


def collate_pairs(samples):
    # Variable geometry dictionaries and source types must never be default-collated.
    return samples


def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        return sample.to(device, non_blocking=sample.is_pinned())
    if isinstance(sample, dict):
        return {key: to_device(value, device) for key, value in sample.items()}
    if isinstance(sample, list):
        return [to_device(value, device) for value in sample]
    return sample
