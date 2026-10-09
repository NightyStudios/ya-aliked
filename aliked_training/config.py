"""Paper settings and explicitly marked reconstruction decisions."""
import json
import math
from pathlib import Path

DEFAULTS = {
    "model": "aliked-n16", "image_size": 800, "batch_size": 2,
    "accumulate_batches": 6, "max_steps": 100000,
    "radius": 2, "top_k": 400, "random_k": 400,
    "gt_threshold": 5.0, "t_det": 0.1, "t_des": 0.1, "t_rel": 1.0,
    "weights": {"rp": 1.0, "pk": 0.5, "ds": 5.0, "re": 1.0},
    "reprojection_norm": 2, "peak_mode": "upstream",
    "learning_rate": 0.0003, "warmup_steps": 500, "adam_betas": [0.9, 0.999],
    "seed": 0, "workers": 4, "device": "cuda", "pairs_per_scene": 10000,
    "megadepth_root": None, "megadepth_manifest": "dataset.json", "excluded_scenes": [],
    "homography_manifest": None, "homography_root": None, "homography_probability": 0.5,
    "homography_tilt": 0.5, "rotation_degrees": 0.0,
    "validation_megadepth_root": None, "validation_homography_manifest": None,
    "validation_interval": 1000, "validation_max_pairs": 200,
    "validation_pairs_per_scene": 50, "validation_seed": 0,
    "validation_megadepth_manifest": "dataset.json", "validation_homography_root": None,
    "validation_selection_metric": "mean_correct_matches",
    "depth_tolerance": 0.05, "inference_detector": "upstream",
    "diagnostics_interval": 100,
    "eval_score_threshold": 0.2, "eval_max_keypoints": 5000,
    "output_dir": "output/aliked-training", "initial_weights": None,
}


def load_config(path):
    config = json.loads(json.dumps(DEFAULTS))
    overrides = json.loads(Path(path).read_text())
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config.update(overrides)
    for name in ("batch_size", "accumulate_batches", "max_steps", "image_size", "radius", "top_k",
                 "validation_interval", "validation_max_pairs", "pairs_per_scene",
                 "validation_pairs_per_scene"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("random_k", "workers", "warmup_steps", "diagnostics_interval", "seed", "validation_seed"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("t_det", "t_des", "t_rel", "learning_rate", "gt_threshold", "depth_tolerance"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config["inference_detector"] not in ("upstream", "training"):
        raise ValueError("inference_detector must be upstream or training")
    if config["validation_selection_metric"] not in ("mean_correct_matches", "mma3", "ms3"):
        raise ValueError("Invalid validation_selection_metric")
    if config["image_size"] <= 2 * config["radius"] + 1:
        raise ValueError("image_size must exceed the DKD window")
    if len(config["adam_betas"]) != 2 or any(not 0 <= b < 1 for b in config["adam_betas"]):
        raise ValueError("adam_betas must contain two values in [0, 1)")
    if config["model"] not in ("aliked-t16", "aliked-n16", "aliked-n32", "aliked-n16rot"):
        raise ValueError("Unknown ALIKED model")
    if config["reprojection_norm"] not in (1, 2) or config["peak_mode"] not in ("upstream", "paper"):
        raise ValueError("Invalid loss convention")
    if config["random_k"] < 0 or config["workers"] < 0:
        raise ValueError("random_k and workers must be non-negative")
    if not 0 <= config["homography_tilt"] <= 1 or not 0 <= config["rotation_degrees"] <= 180:
        raise ValueError("Invalid homography tilt or rotation range")
    if not 0 < config["homography_probability"] < 1:
        raise ValueError("homography_probability must be strictly between 0 and 1")
    if type(config["eval_max_keypoints"]) is not int or config["eval_max_keypoints"] < 1:
        raise ValueError("eval_max_keypoints must be a positive integer")
    if not 0 <= config["eval_score_threshold"] <= 1:
        raise ValueError("eval_score_threshold must be in [0, 1]")
    if set(config["weights"]) != {"rp", "pk", "ds", "re"}:
        raise ValueError("weights must specify rp, pk, ds and re")
    if any(not math.isfinite(value) or value < 0 for value in config["weights"].values()):
        raise ValueError("Loss weights must be finite and non-negative")
    return config
