import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import registry
from app.detectors.sift_cv import _run_cv_sift
from app.image_io import decode_image_b64
from app.main import app
from app.schemas import Detection


def _scene(seed=0, shape=(300, 400, 3)):
    """Синтетическая сцена из кругов и прямоугольников: стабильные углы/блобы."""
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    img = np.zeros(shape, np.uint8)
    for _ in range(60):
        color = tuple(int(c) for c in rng.integers(30, 255, 3))
        cv2.circle(img, (int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))),
                   int(rng.integers(3, 15)), color, -1)
        cv2.rectangle(img, (int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))),
                      (int(rng.integers(0, w)), int(rng.integers(0, h))), color, 2)
    return img


def _png_b64(img):
    import base64
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class SiftDetectorTests(unittest.TestCase):
    def test_registered_and_available(self):
        meta = registry.DETECTORS["sift_cv"]
        self.assertTrue(meta["available"])
        self.assertFalse(meta["hidden"])
        self.assertEqual(meta["group"], "SIFT")
        self.assertIs(registry.get_detector("sift_cv"), _run_cv_sift)

    def test_output_contract(self):
        pts, desc, norm = _run_cv_sift(_scene())
        self.assertEqual(norm, "l2")
        self.assertGreater(len(pts), 0)
        self.assertEqual(pts.dtype, np.float32)
        self.assertEqual(pts.shape, (len(pts), 2))
        self.assertEqual(desc.dtype, np.float32)
        self.assertEqual(desc.shape, (len(pts), 128))

    def test_n_features_limits_count(self):
        pts, _, _ = _run_cv_sift(_scene(), n_features=20)
        self.assertLessEqual(len(pts), 20)

    def test_unknown_params_are_ignored(self):
        pts, desc, _ = _run_cv_sift(_scene(), foo="bar")
        self.assertEqual(len(pts), len(desc))

    def test_blank_image_gives_empty_detection(self):
        blank = np.full((64, 64, 3), 127, np.uint8)
        pts, desc, norm = _run_cv_sift(blank)
        self.assertEqual(pts.shape, (0, 2))
        self.assertEqual(desc.shape, (0, 128))
        self.assertEqual(desc.dtype, np.float32)
        det = Detection.from_arrays(pts, desc, norm)
        self.assertEqual(det.desc_shape, (0, 128))
        self.assertEqual(det.desc_dtype, "float32")

    def test_grayscale_input(self):
        gray = cv2.cvtColor(_scene(), cv2.COLOR_BGR2GRAY)
        # decode_image_b64 всегда отдаёт BGR; прогоняем серую через тот же путь
        img = decode_image_b64(_png_b64(gray))
        pts, desc, _ = _run_cv_sift(img)
        self.assertEqual(desc.shape[1], 128)
        self.assertEqual(len(pts), len(desc))


class SiftPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_methods_lists_sift(self):
        r = self.client.get("/api/methods")
        self.assertEqual(r.status_code, 200)
        meta = r.json()["detector_meta"]["sift_cv"]
        self.assertTrue(meta["available"])
        self.assertEqual(meta["group"], "SIFT")

    def test_detect_endpoint(self):
        r = self.client.post("/api/detect", json={
            "method": "sift_cv",
            "image": {"image_b64": _png_b64(_scene())},
        })
        self.assertEqual(r.status_code, 200)
        det = r.json()["detection"]
        self.assertEqual(det["norm"], "l2")
        self.assertEqual(det["desc_dtype"], "float32")
        n, d = det["desc_shape"]
        self.assertEqual(d, 128)
        self.assertEqual(n, len(det["points"]))

    def _pipeline(self, img_a, img_b, **extra):
        return self.client.post("/api/pipeline", json={
            "method": "sift_cv", "matcher": "bf",
            "image_a": {"image_b64": _png_b64(img_a)},
            "image_b": {"image_b64": _png_b64(img_b)},
            **extra,
        })

    def test_pipeline_matches_shifted_image(self):
        img = _scene(seed=1)
        shift = np.float32([[1, 0, 12], [0, 1, 7]])
        shifted = cv2.warpAffine(img, shift, (img.shape[1], img.shape[0]))

        r = self._pipeline(img, shifted)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["det_a"]["norm"], "l2")
        self.assertGreater(len(body["matches"]), 10)

        pa, pb = body["det_a"]["points"], body["det_b"]["points"]
        good = 0
        for m in body["matches"]:
            ax, ay = pa[m["a_idx"]]
            bx, by = pb[m["b_idx"]]
            if abs(bx - ax - 12) < 2 and abs(by - ay - 7) < 2:
                good += 1
        self.assertGreater(good / len(body["matches"]), 0.8)

    def test_pipeline_ratio_test_and_ransac(self):
        img = _scene(seed=2)
        r = self._pipeline(img, img, matcher_params={"ratio": 0.75},
                           ransac=True)
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()["matches"]), 10)

    def test_pipeline_blank_image_does_not_crash(self):
        blank = np.full((64, 64, 3), 127, np.uint8)
        r = self._pipeline(_scene(), blank)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["matches"], [])
        r = self._pipeline(blank, blank, matcher_params={"ratio": 0.75})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["matches"], [])


if __name__ == "__main__":
    unittest.main()
