"""Paper settings and explicitly marked reconstruction decisions."""
import json
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
    "validation_interval": 1000, "validation_max_pairs": 100,
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
                 "validation_interval", "validation_max_pairs", "pairs_per_scene"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("t_det", "t_des", "t_rel", "learning_rate"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
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
    if config["warmup_steps"] < 0 or config["eval_max_keypoints"] < 1:
        raise ValueError("Invalid warmup_steps or eval_max_keypoints")
    if set(config["weights"]) != {"rp", "pk", "ds", "re"}:
        raise ValueError("weights must specify rp, pk, ds and re")
    if any(value < 0 for value in config["weights"].values()):
        raise ValueError("Loss weights must be non-negative")
    return config
