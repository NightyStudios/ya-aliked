import torch

from aliked_training.detector import (
    InferenceDetector, TrainingDetector, integer_patches, point_nms,
)
from aliked_training.vendor.soft_detect import DKD


def test_point_nms_stable_ties_and_original_order():
    points = torch.tensor([[0., 0.], [0.5, 0.], [4., 0.]])
    scores = torch.tensor([0.5, 0.5, 0.9])
    assert point_nms(points, scores, 1).tolist() == [0, 2]


def test_point_nms_greedy_chain_and_strict_radius():
    # B is suppressed by A, so B must not suppress C.
    points = torch.tensor([[0., 0.], [0.75, 0.], [1.5, 0.], [2.5, 0.]])
    scores = torch.tensor([4., 3., 2., 1.])
    assert point_nms(points, scores, 1).tolist() == [0, 2, 3]
    assert point_nms(points[:0], scores[:0], 1).shape == (0,)


def test_point_nms_selected_values_keep_gradients():
    points = torch.tensor([[0., 0.], [0.5, 0.], [4., 0.]], requires_grad=True)
    scores = torch.tensor([0.9, 0.5, 0.7], requires_grad=True)
    keep = point_nms(points, scores, 1)
    (points[keep].sum() + scores[keep].sum()).backward()
    torch.testing.assert_close(scores.grad, torch.tensor([1., 0., 1.]))
    torch.testing.assert_close(points.grad, torch.tensor([[1., 1.], [0., 0.], [1., 1.]]))


def test_integer_patches_accumulates_overlapping_gradients():
    fmap = torch.arange(25., requires_grad=True).reshape(1, 5, 5)
    fmap.retain_grad()
    centers = torch.tensor([[2, 2], [3, 2]])
    patches = integer_patches(fmap, centers, 3)
    reference = torch.nn.functional.unfold(fmap[None], 3, padding=1)
    expected = reference[0, :, torch.tensor([12, 13])].T.reshape(2, 1, 3, 3)
    torch.testing.assert_close(patches, expected)
    patches.sum().backward()
    expected_grad = torch.zeros_like(fmap)
    expected_grad[:, 1:4, 1:4] += 1
    expected_grad[:, 1:4, 2:5] += 1
    torch.testing.assert_close(fmap.grad, expected_grad)


def test_inference_low_scores_use_public_fallback():
    score_map = torch.full((1, 1, 13, 13), 0.01)
    score_map[0, 0, 6, 6] = 0.1
    inference = InferenceDetector(radius=2, scores_th=0.2)(score_map)
    training = TrainingDetector(radius=2, scores_th=0.2)(
        score_map, add_random=False, threshold=True,
    )
    assert inference["num_detected"] == [1]
    assert training["num_detected"] == [0]
    torch.testing.assert_close(inference["keypoints"][0], torch.zeros(1, 2), atol=1e-6, rtol=0)


def test_inference_matches_public_threshold_order_and_limit():
    generator = torch.Generator().manual_seed(51)
    score_map = torch.rand((2, 1, 24, 24), generator=generator)
    for limit in (3, 5000):
        result = InferenceDetector(radius=2, scores_th=0.2, n_limit=limit)(score_map)
        expected = DKD(radius=2, scores_th=0.2, n_limit=limit)(score_map)
        for key, values in zip(("keypoints", "scores", "score_dispersity"), expected):
            for actual, reference in zip(result[key], values):
                torch.testing.assert_close(actual, reference)
        assert result["num_detected"] == [len(p) for p in expected[0]]


def test_inference_uniform_scores_can_be_empty():
    result = InferenceDetector()(torch.zeros((1, 1, 13, 13)))
    assert result["num_detected"] == [0]
