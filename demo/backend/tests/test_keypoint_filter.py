import unittest

import numpy as np

from app.keypoint_filter import filter_by_mask


class KeypointFilterTests(unittest.TestCase):
    def test_margin_zero_and_class_zero(self):
        mask = np.array([[0, 1, 2]], np.uint8)
        self.assertEqual(filter_by_mask(mask, [[0, 0], [1, 0], [2, 0]], [0, 1], 0)["keep"],
                         [False, False, True])

    def test_subpixel_margin_inclusive(self):
        mask = np.array([[5, 5, 5, 2]], np.uint8)
        self.assertTrue(filter_by_mask(mask, [[2.25, 0]], [5], .75)["keep"][0])
        self.assertFalse(filter_by_mask(mask, [[2.25, 0]], [5], .74)["keep"][0])

    def test_euclidean_distance_not_square_window(self):
        mask = np.full((4, 4), 5, np.uint8)
        mask[2, 2] = 2
        self.assertFalse(filter_by_mask(mask, [[1, 1]], [5], 1)["keep"][0])
        self.assertTrue(filter_by_mask(mask, [[1, 1]], [5], np.sqrt(2))["keep"][0])

    def test_all_classes_excluded_image_exterior_is_not_allowed(self):
        result = filter_by_mask(np.zeros((3, 3), np.uint8), [[0, 0], [2, 2]], [0], 1e9)
        self.assertEqual(result["keep"], [False, False])

    def test_empty_exclusions_and_points(self):
        mask = np.zeros((3, 3), np.uint8)
        self.assertEqual(filter_by_mask(mask, [[1, 1]], [], 0)["keep"], [True])
        self.assertEqual(filter_by_mask(mask, [], [0], 5)["keep"], [])

    def test_rounding_at_image_edge(self):
        self.assertEqual(filter_by_mask(np.array([[0, 1]], np.uint8), [[1.9, .9]], [1], 0)["keep"], [False])

    def test_invalid_coordinates_or_margin(self):
        mask = np.zeros((3, 3), np.uint8)
        for points, margin in [([[3, 0]], 0), ([[float('nan'), 0]], 0), ([[0, 0]], -1)]:
            with self.assertRaises(ValueError):
                filter_by_mask(mask, points, [0], margin)


if __name__ == '__main__':
    unittest.main()
