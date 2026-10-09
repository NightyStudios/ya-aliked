import json

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from aliked_training.data import DISKMegaDepth, HomographicPairs, MixedPairs, collate_pairs, to_device


@pytest.fixture
def disk_root(tmp_path):
    for folder in ("images", "depth", "calib"):
        (tmp_path / folder).mkdir()
    # Noninteger resize ratio exposes the legacy nearest-neighbour offset.
    depth = np.tile(np.arange(5, dtype=np.float32), (3, 1))
    for name in ("a", "b"):
        Image.fromarray(np.full((3, 5, 3), 128, dtype=np.uint8)).save(tmp_path / "images" / f"{name}.png")
        with h5py.File(tmp_path / "depth" / f"{name}.h5", "w") as file:
            file["depth"] = depth
        with h5py.File(tmp_path / "calib" / f"calibration_{name}.h5", "w") as file:
            file["K"] = np.array([[10, 0, 2.5], [0, 10, 1.5], [0, 0, 1]], dtype=np.float32)
            file["R"] = np.eye(3, dtype=np.float32)
            file["T"] = np.zeros(3, dtype=np.float32)
    scene = {"images": ["a.png", "b.png"], "tuples": [[0, 1]],
             "image_path": "images", "depth_path": "depth", "calib_path": "calib"}
    (tmp_path / "dataset.json").write_text(json.dumps({"scene": scene}))
    return tmp_path


def test_depth_resize_matches_pixel_centres_and_preserves_holes(disk_root):
    sample = DISKMegaDepth(disk_root, size=3, augment=False, depth_tolerance=0.2)[0]
    for direction in ("warp01", "warp10"):
        geometry = sample[direction]
        torch.testing.assert_close(geometry["depth0"], torch.tensor([[0., 2., 4.]]).expand(3, 3))
        assert geometry["depth_tolerance"] == 0.2
        torch.testing.assert_close(geometry["K0"], torch.tensor([[6., 0., 1.5], [0., 10., 1.5], [0., 0., 1.]]))
        torch.testing.assert_close(geometry["T01"], torch.eye(4))
    assert DISKMegaDepth(disk_root, size=3, augment=False)[0]["warp01"]["depth_tolerance"] == 0.05


@pytest.mark.parametrize("tolerance", [0, -0.1, float("nan"), float("inf")])
def test_invalid_depth_tolerance(disk_root, tolerance):
    with pytest.raises(ValueError, match="depth_tolerance"):
        DISKMegaDepth(disk_root, depth_tolerance=tolerance)


@pytest.fixture
def homographic(tmp_path):
    pixels = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    Image.fromarray(pixels).save(tmp_path / "image.png")
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps({"image0": "image.png"}) + "\n")
    return HomographicPairs(manifest, size=24, augment=True, seed=17, rotation=20)


def test_direct_sampling_is_repeatable_and_preserves_rng(homographic):
    rng_before = torch.random.get_rng_state().clone()
    first, second = homographic[0], homographic[0]
    torch.testing.assert_close(first["image1"], second["image1"])
    torch.testing.assert_close(first["warp01"]["H"], second["warp01"]["H"])
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_mixed_draws_have_distinct_repeatable_augmentations(homographic):
    class Perspective:
        def __len__(self):
            return 32

        def set_epoch(self, epoch):
            pass

        def __getitem__(self, index):
            return {"source": "megadepth"}

    mixed = MixedPairs(Perspective(), homographic, homography_probability=0.999999, seed=3)
    first, other = mixed[0], mixed[1]
    assert first["source"] == other["source"] == "homography"
    assert not torch.equal(first["image1"], other["image1"])
    assert not torch.equal(first["warp01"]["H"], other["warp01"]["H"])
    torch.testing.assert_close(first["image1"], mixed[0]["image1"])
    mixed.set_epoch(1)
    assert not torch.equal(first["image1"], mixed[0]["image1"])
    mixed.set_epoch(0)
    torch.testing.assert_close(first["image1"], mixed[0]["image1"])


@pytest.mark.parametrize("pinned", [False, True])
def test_transfer_preserves_list_collation_and_uses_pinning(monkeypatch, pinned):
    calls = []
    original_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        calls.append(kwargs["non_blocking"])
        return original_to(tensor, *args, **kwargs)

    # Simulate pinned storage without requiring a CUDA allocator in CPU CI.
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: pinned)
    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    samples = [{"image0": torch.ones(3, 2, 2), "warp01": {"H": torch.eye(3), "mode": "homo"}}]
    assert collate_pairs(samples) is samples
    result = to_device(samples, "cpu")
    assert isinstance(result, list)
    assert result[0]["warp01"]["mode"] == "homo"
    assert calls == [pinned, pinned]


def test_cli_real_manifests_and_cross_source_overlap(disk_root, homographic, tmp_path):
    from aliked_training.cli import datasets_from_config, data_provenance
    from aliked_training.config import DEFAULTS

    for name in ("holdout0.png", "holdout1.png"):
        Image.new("RGB", (24, 24)).save(tmp_path / name)
    manifest = tmp_path / "validation.jsonl"
    row = {"image0": "holdout0.png", "image1": "holdout1.png", "aligned": True}
    manifest.write_text(json.dumps(row) + "\n")
    cfg = dict(DEFAULTS, megadepth_root=str(disk_root), image_size=24,
               homography_manifest=str(homographic.manifest),
               validation_homography_manifest=str(manifest))
    train, validation = datasets_from_config(cfg)
    before = data_provenance(train, validation, cfg)
    assert before["homography"]["indices"] == [0]
    assert before["train/megadepth"]["scenes"] == ["scene"]
    manifest.write_text(json.dumps(row, indent=None) + "\n\n")
    after = data_provenance(train, validation, cfg)
    assert before["homography"]["sha256"] != after["homography"]["sha256"]
    # Holdout homography references a training MegaDepth image, even though
    # the two homography manifests themselves have no paths in common.
    row["image0"] = str(disk_root / "images" / "a.png")
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="images overlap"):
        datasets_from_config(cfg)
