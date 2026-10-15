import unittest

import numpy as np

from urban3d_eval.dataset import SOURCE_VISIBLE, GroundTruth
from urban3d_eval.metrics import pair_metrics, reference_homography, sample_forward

H, W = 64, 96


def make_gt(matrix=None, status=SOURCE_VISIBLE):
    """GT of a planar transform `matrix` (3x3); everything visible by default."""
    matrix = np.eye(3) if matrix is None else matrix
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float64)
    ones = np.ones_like(cols)
    moved = np.einsum('ij,jhw->ihw', matrix, np.stack([cols, rows, ones]))
    forward = np.stack([moved[0] / moved[2], moved[1] / moved[2]], axis=-1).astype(np.float32)
    return GroundTruth(forward_xy=forward, source_status=np.full((H, W), status, np.uint8),
                       roll_deg=0.0, tilt_deg=0.0)


SHIFT = np.array([[1, 0, 5.5], [0, 1, -3.25], [0, 0, 1]])


class SampleForwardTests(unittest.TestCase):
    def test_bilinear_on_fractional_points(self):
        gt = make_gt(SHIFT)
        points = np.array([[10.3, 20.7], [0, 0], [W - 1, H - 1]])
        projected, valid = sample_forward(gt, points)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(projected, points + [5.5, -3.25], atol=1e-4)

    def test_outside_and_invisible_are_invalid(self):
        gt = make_gt(SHIFT)
        gt.source_status[10, 10] = 3                       # occluded
        gt.forward_xy[11, 11] = np.nan                     # NoData
        gt.source_status[11, 11] = 0
        _, valid = sample_forward(gt, np.array([[10, 10], [11, 11], [-1, 5], [5, H], [20, 20]]))
        self.assertEqual(valid.tolist(), [False, False, False, False, True])

    def test_neighbour_occlusion_does_not_poison_interpolation(self):
        gt = make_gt(SHIFT)
        gt.source_status[21, 21] = 3
        gt.forward_xy[21, 21] = [1000, 1000]               # garbage behind the occluder
        projected, valid = sample_forward(gt, np.array([[20.2, 20.2]]))
        self.assertTrue(valid[0])
        # renormalised weights are not exact for a dropped neighbour, but stay sub-0.05 px
        np.testing.assert_allclose(projected[0], [20.2 + 5.5, 20.2 - 3.25], atol=0.05)


class ReferenceHomographyTests(unittest.TestCase):
    def test_recovers_planar_homography(self):
        truth = np.array([[0.98, 0.1, 7.0], [-0.08, 1.02, -4.0], [1e-4, 2e-4, 1.0]])
        reference, ratio = reference_homography(make_gt(truth), step=4)
        self.assertGreater(ratio, 0.99)
        np.testing.assert_allclose(reference / reference[2, 2], truth, atol=1e-3)

    def test_no_visible_points(self):
        reference, ratio = reference_homography(make_gt(status=3))
        self.assertIsNone(reference)
        self.assertEqual(ratio, 0.0)


class PairMetricsTests(unittest.TestCase):
    thresholds = (1, 2, 3)

    def setUp(self):
        rng = np.random.default_rng(0)
        self.gt = make_gt(SHIFT)
        self.reference, _ = reference_homography(self.gt, step=4)
        self.points_a = rng.uniform([2, 2], [W - 8, H - 4], (40, 2)).astype(np.float32)
        self.matches = np.column_stack((np.arange(40), np.arange(40)))

    def test_perfect_matches(self):
        points_b = self.points_a + np.float32([5.5, -3.25])
        r = pair_metrics(self.points_a, points_b, self.matches, self.gt, self.reference,
                         (H, W, 3), self.thresholds)
        self.assertEqual(r['mma'], [1.0, 1.0, 1.0])
        self.assertEqual(r['mha'], [1.0, 1.0, 1.0])
        self.assertEqual(r['scored_matches'], 40)

    def test_mma_counts_error_thresholds(self):
        points_b = self.points_a + np.float32([5.5, -3.25])
        points_b[:10] += [1.5, 0]        # 10 matches off by 1.5 px
        points_b[10:20] += [0, 2.5]      # 10 matches off by 2.5 px
        r = pair_metrics(self.points_a, points_b, self.matches, self.gt, self.reference,
                         (H, W, 3), self.thresholds)
        np.testing.assert_allclose(r['mma'], [0.5, 0.75, 1.0], atol=1e-6)

    def test_matches_without_gt_are_not_scored(self):
        self.gt.source_status[:, : W // 2] = 3
        points_b = self.points_a + np.float32([5.5, -3.25])
        r = pair_metrics(self.points_a, points_b, self.matches, self.gt, self.reference,
                         (H, W, 3), self.thresholds)
        self.assertEqual(r['scored_matches'], int((self.points_a[:, 0] >= W // 2 - 0.5).sum()))
        self.assertEqual(r['mma'], [1.0, 1.0, 1.0])

    def test_too_few_matches_scores_zero(self):
        points_b = self.points_a + np.float32([5.5, -3.25])
        r = pair_metrics(self.points_a, points_b, self.matches[:3], self.gt, self.reference,
                         (H, W, 3), self.thresholds)
        self.assertEqual(r['mha'], [0.0, 0.0, 0.0])
        self.assertEqual(r['mma'], [1.0, 1.0, 1.0])
        r = pair_metrics(self.points_a, self.points_a, np.empty((0, 2), int), self.gt,
                         self.reference, (H, W, 3), self.thresholds)
        self.assertEqual(r['mma'], [0.0, 0.0, 0.0])
        self.assertIsNone(r['median_error_px'])

    def test_wrong_homography_fails_mha(self):
        points_b = self.points_a + np.float32([5.5, -3.25]) + np.float32([20, 20])
        r = pair_metrics(self.points_a, points_b, self.matches, self.gt, self.reference,
                         (H, W, 3), self.thresholds)
        self.assertEqual(r['mha'], [0.0, 0.0, 0.0])
        self.assertEqual(r['mma'], [0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
