import copy
import json

import pytest
import torch

from aliked_training import cli
from aliked_training.config import DEFAULTS, load_config


class SyntheticPairs(torch.utils.data.Dataset):
    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 6

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(10 + 100 * self.epoch + index)
        image = torch.rand(3, 64, 64, generator=generator)
        warp = {"mode": "homo", "H": torch.eye(3), "shape1": (64, 64)}
        return {"image0": image, "image1": image.clone(), "warp01": warp,
                "warp10": warp, "source": "homography"}


def configuration(path, steps):
    return dict(copy.deepcopy(DEFAULTS), model="aliked-t16", image_size=64,
                device="cpu", workers=0, batch_size=2, accumulate_batches=2,
                max_steps=steps, top_k=8, random_k=4, eval_max_keypoints=16,
                validation_max_pairs=2, validation_interval=1,
                diagnostics_interval=1, output_dir=str(path))


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_real_model_resume_matches_continuous_across_epoch(tmp_path, monkeypatch):
    # Six samples => three microbatches/epoch. Accumulation=2 crosses an epoch.
    # Exercise the actual model, losses, optimizer, validation, sampler and RNG.
    monkeypatch.setattr(cli, "datasets_from_config", lambda _: (SyntheticPairs(), {"toy": SyntheticPairs()}))
    monkeypatch.setattr(cli, "data_provenance", lambda *args: {"toy": {"sha256": "fixed"}})
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    initial = tmp_path / "initial.pth"
    torch.save(cli.TrainableALIKED(configuration(continuous, 3)).state_dict(), initial)
    cli.train(dict(configuration(continuous, 3), initial_weights=str(initial)))
    cli.train(dict(configuration(interrupted, 1), initial_weights=str(initial)))
    initial.unlink()  # A full checkpoint must not require its old initialization file.
    cli.train(dict(configuration(interrupted, 3), initial_weights=str(initial)), interrupted / "last.pt")
    a = torch.load(continuous / "last.pt", weights_only=False)
    b = torch.load(interrupted / "last.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "step", "epoch", "batch_index", "torch_rng", "best_metric"):
        assert_nested_equal(a[key], b[key])
    records = [json.loads(line) for line in (interrupted / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2, 3]
    assert records[-1]["samples_seen"] == 12
    assert records[-1]["diagnostics/score_grad_norm/ds"] > 0
    assert (interrupted / "data-provenance.json").exists()
    assert (interrupted / "best-model.pth").exists()


@pytest.mark.parametrize("field,value", [("depth_tolerance", 0), ("t_rel", float("nan")),
                                        ("batch_size", 1.5), ("diagnostics_interval", -1),
                                        ("inference_detector", "typo"),
                                        ("validation_selection_metric", "accuracy")])
def test_invalid_config(tmp_path, field, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({field: value}))
    with pytest.raises(ValueError):
        load_config(path)


def test_shipped_configs_are_loadable():
    baseline = load_config("configs/aliked/paper.json")
    alternative = load_config("configs/aliked/code-temperature.json")
    assert baseline["t_rel"] == 1
    assert alternative["t_rel"] == .1
    assert baseline["validation_pairs_per_scene"] == 50


def test_inspect_does_not_construct_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection constructed a model")
    monkeypatch.setattr(cli, "TrainableALIKED", forbidden)
    assert cli.inspect(configuration(tmp_path, 1))["training_started"] is False
