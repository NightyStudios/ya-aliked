import random
import torch
import pytest
from aliked_training.geometry import warp_points
from aliked_training.validation import pair_statistics, validate, validation_indices


def se3(depth0, depth1, translation=0):
    transform = torch.eye(4)
    transform[0, 3] = translation
    return dict(mode="se3", depth0=depth0, depth1=depth1, K0=torch.eye(3),
                K1=torch.eye(3), T01=transform)


def homo(dx=0):
    matrix = torch.eye(3)
    matrix[0, 2] = dx
    return dict(mode="homo", H=matrix, shape1=(16, 16))


def test_depth_hole_is_unknown_not_incorrect():
    depth = torch.ones(16, 16)
    depth[8, 8] = 0
    points = torch.tensor([[2., 2.], [8., 8.]])
    stats = pair_statistics(points, points, torch.eye(2), torch.eye(2),
                            se3(depth, depth), se3(depth, depth))
    assert (stats["n_putative"], stats["n_evaluable"], stats["n_correct"]) == (2, 1, 1)
    warp = warp_points(points, se3(depth, depth))
    assert warp.evaluable.tolist() == [True, False]
    assert warp.unknown.tolist() == [False, True]


def test_outside_is_known_negative_even_with_unknown_reverse():
    points = torch.tensor([[2., 2.]])
    depth = torch.ones(16, 16)
    stats = pair_statistics(points, points, torch.ones(1, 1), torch.ones(1, 1),
                            se3(depth, depth, 30), se3(depth*0, depth))
    assert stats["n_evaluable"] == 1
    assert stats["n_correct"] == 0


def test_reverse_geometry_required():
    points = torch.tensor([[2., 2.]])
    desc = torch.ones(1, 1)
    stats = pair_statistics(points, points, desc, desc, homo(), homo(4))
    assert stats["n_evaluable"] == 1
    assert stats["n_correct"] == 0
    depth = torch.ones(16, 16)
    stats = pair_statistics(points, points, desc, desc,
                            se3(depth, depth), se3(depth*0, depth))
    assert stats["n_evaluable"] == 0


class PairDataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return dict(image0=torch.zeros(3, 16, 16), image1=torch.zeros(3, 16, 16),
                    warp01=homo(), warp10=homo())


class FixedModel(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def forward(self, images, **kwargs):
        points = torch.stack((torch.arange(self.count).float()+2,
                              torch.ones(self.count)*2), dim=-1)
        return dict(keypoints=[points / 15 * 2 - 1], descriptors=[torch.eye(self.count)])


def evaluate(count, dataset=None, **config):
    return validate(FixedModel(count), {"test": dataset or PairDataset()},
                    dict(validation_max_pairs=3, seed=42, **config), "cpu")


def test_default_selection_rewards_more_correct_matches():
    one, many = evaluate(1), evaluate(8)
    assert one["test/mma3"] == many["test/mma3"] == 1
    assert many["selection_metric"] > one["selection_metric"]
    assert many["test/ms3"] == many["test/repeatability3"] == 1


def test_empty_detector_gets_zero_not_false_perfect_precision():
    result = evaluate(0)
    assert result["test/mma3"] is None
    assert result["selection_metric"] == 0


def test_all_unknown_cannot_select_checkpoint():
    class Unknown(PairDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            depth = torch.zeros(16, 16)
            sample.update(warp01=se3(depth, depth), warp10=se3(depth, depth))
            return sample
    with pytest.raises(ValueError, match="no evaluable"):
        evaluate(2, Unknown())


def test_validation_preserves_rng_and_model_mode():
    torch.manual_seed(29)
    random.seed(30)
    state, python_state = torch.random.get_rng_state(), random.getstate()
    model = FixedModel(2)
    validate(model, {"test": PairDataset()}, {"validation_max_pairs": 2}, "cpu")
    assert model.training
    assert torch.equal(state, torch.random.get_rng_state())
    assert python_state == random.getstate()


def test_indices_fixed_stratified_unique():
    class Scenes:
        counts = [10, 10, 10]
        ends = [10, 20, 30]
        def __len__(self):
            return 30
    indices = validation_indices(Scenes(), 6, 42)
    assert indices == validation_indices(Scenes(), 6, 42)
    assert len(set(indices)) == 6
    assert sorted(i // 10 for i in indices) == [0, 0, 1, 1, 2, 2]
    assert validation_indices(Scenes(), 6, 41) != indices


def test_unknown_source_cannot_silently_change_source_weights():
    class Unknown(PairDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            depth = torch.zeros(16, 16)
            sample.update(warp01=se3(depth, depth), warp10=se3(depth, depth))
            return sample
    with pytest.raises(ValueError, match="source megadepth has no evaluable"):
        validate(FixedModel(2), {"homography": PairDataset(), "megadepth": Unknown()},
                 {"validation_max_pairs": 2}, "cpu")
