"""Regression checks for sparse supervision and optional training diagnostics."""
import math

import torch
from torch.nn import functional as F

from aliked_training.losses import pair_loss, reliable_loss, sparse_nre


def test_sparse_nre_denominator_includes_unmatched_probes():
    # Only index 0 matches, but index 1 remains a competing descriptor.
    d0 = torch.eye(2, requires_grad=True)
    d1 = torch.eye(2, requires_grad=True)
    ids = torch.tensor([0])
    loss = sparse_nre(d0, d1, ids, ids, 1.)
    torch.testing.assert_close(loss, torch.tensor(math.log1p(math.exp(-1))))
    loss.backward()
    assert d0.grad[1].abs().sum() > 0
    assert d1.grad[1].abs().sum() > 0


def test_reliability_only_trains_scores_with_detached_targets():
    d0 = torch.eye(2, requires_grad=True)
    d1 = torch.tensor([[1., 0.], [0.6, 0.8]], requires_grad=True)
    s0 = torch.tensor([0.4, 0.8], requires_grad=True)
    s1 = torch.tensor([0.7, 0.3], requires_grad=True)
    ids = torch.arange(2)
    loss, targets = reliable_loss(d0, d1, s0, s1, ids, ids, 0.3, return_targets=True)
    assert not targets.requires_grad
    loss.backward()
    assert d0.grad is None and d1.grad is None
    for score in (s0, s1):
        assert torch.isfinite(score.grad).all() and score.grad.abs().sum() > 0


def test_empty_matching_losses_are_finite_differentiable_zero():
    d0 = torch.randn(3, 4, requires_grad=True)
    d1 = torch.randn(2, 4, requires_grad=True)
    s0 = torch.rand(3, requires_grad=True)
    s1 = torch.rand(2, requires_grad=True)
    ids = torch.empty(0, dtype=torch.long)
    loss = sparse_nre(d0, d1, ids, ids, 0.1) + reliable_loss(d0, d1, s0, s1, ids, ids, 1.)
    assert loss.item() == 0
    loss.backward()
    for value in (d0, d1, s0, s1):
        torch.testing.assert_close(value.grad, torch.zeros_like(value))


def example():
    def pred(points, descriptors):
        return {
            'score_map': torch.ones(1, 1, 9, 9, requires_grad=True),
            'keypoints': [torch.tensor(points, requires_grad=True)],
            'descriptors': [F.normalize(torch.tensor(descriptors, requires_grad=True), dim=-1)],
            'scores': [torch.tensor([0.4, 0.8], requires_grad=True)],
            'score_dispersity': [torch.tensor([0.2], requires_grad=True)],
            'num_detected': [1],
            'offsets': [torch.tensor([[[0., 0.], [20., 0.]], [[0., 0.], [0., 0.]]])],
        }
    p0 = pred([[-0.5, -0.5], [0.5, 0.5]], [[1., 0.], [0., 1.]])
    p1 = pred([[-0.45, -0.5], [0.5, 0.45]], [[1., 0.], [0.6, 0.8]])
    warp = {'mode': 'homo', 'H': torch.eye(3), 'shape1': (9, 9)}
    sample = {'image0': torch.zeros(3, 9, 9), 'image1': torch.zeros(3, 9, 9),
              'warp01': warp, 'warp10': warp, 'source': 'homography'}
    config = {'gt_threshold': 1., 'reprojection_norm': 2, 't_des': 0.1,
              't_rel': 1., 'weights': {'rp': 1., 'pk': 0.5, 'ds': 5., 're': 1.}}
    return p0, p1, [sample], config


def test_diagnostics_preserve_values_rng_and_gradient_paths():
    p0, p1, batch, config = example()
    state = torch.random.get_rng_state().clone()
    plain = pair_loss(p0, p1, batch, config)
    logged = pair_loss(p0, p1, batch, config, diagnostics=True)
    assert set(plain) == {'rp', 'pk', 'ds', 're', 'total', 'matches'}
    for key in plain:
        torch.testing.assert_close(plain[key], logged[key], rtol=0, atol=0)
    torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
    diag = logged['diagnostics']
    assert all(value.ndim == 0 and not value.requires_grad for value in diag.values())
    assert diag['matches_mean'] == 2 and diag['empty_pair_fraction'] == 0
    assert diag['detected_points_mean'] == 1 and diag['random_points_mean'] == 1
    assert diag['visible_ratio'] == 1 and diag['unknown_ratio'] == 0
    assert diag['sddh_outside_ratio'] == 0.25
    assert diag['reliability_target_count'] == 4
    assert diag['reliability_target_std'] > 0
    for key in ('rp', 'pk', 'ds', 're'):
        torch.testing.assert_close(diag[f'source/homography/{key}'], plain[key])
    inputs = tuple(p[field][0] for p in (p0, p1)
                   for field in ('keypoints', 'descriptors', 'scores', 'score_dispersity'))
    g0 = torch.autograd.grad(plain['total'], inputs, retain_graph=True)
    g1 = torch.autograd.grad(logged['total'], inputs)
    for a, b in zip(g0, g1):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_pair_loss_without_correspondences_and_empty_predictions():
    p0, p1, batch, config = example()
    for pred in (p0, p1):
        for field in ('keypoints', 'descriptors', 'scores', 'score_dispersity', 'offsets'):
            pred[field] = [pred[field][0][:0]]
        pred['num_detected'] = [0]
    loss = pair_loss(p0, p1, batch, config, diagnostics=True)
    assert loss['total'].item() == 0 and loss['matches'] == 0
    assert loss['diagnostics']['empty_pair_fraction'] == 1
    assert all(torch.isfinite(v) for v in loss['diagnostics'].values())
    loss['total'].backward()
    assert p0['score_map'].grad is not None
