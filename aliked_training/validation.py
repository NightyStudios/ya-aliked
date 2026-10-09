"""Fixed holdout evaluation; matching proxies, not pose/homography accuracy.

MMA uses only matches with evaluable symmetric geometry. MS and repeatability
use min(n_visible0, n_visible1) co-visible keypoints as their denominator.
Metrics are pooled across pairs; source selection scores have equal weight.
"""
import random
import torch
from .data import collate_pairs, to_device
from .geometry import warp_points, mutual_pairs


def validation_indices(dataset, max_pairs, seed=0):
    """Fixed random subset, interleaved across MegaDepth scenes when available."""
    if max_pairs < 1:
        raise ValueError("validation_max_pairs must be positive")
    rng = random.Random(seed)
    limit = min(max_pairs, len(dataset))
    if hasattr(dataset, "counts") and hasattr(dataset, "ends"):
        groups = []
        for count, end in zip(dataset.counts, dataset.ends):
            # Sampling only the possible budget also avoids huge permutations.
            groups.append(rng.sample(range(end-count, end), min(count, limit)))
        rng.shuffle(groups)
        indices = []
        for row in range(limit):
            for group in groups:
                if row < len(group):
                    indices.append(group[row])
                    if len(indices) == limit:
                        return indices
        return indices
    return rng.sample(range(len(dataset)), limit)


def pair_statistics(k0, k1, d0, d1, warp01, warp10, threshold=3):
    """Counts with explicit unknown-depth handling and bidirectional checks."""
    w0, w1 = warp_points(k0, warp01), warp_points(k1, warp10)
    counts = {
        "n_keypoints0": len(k0), "n_keypoints1": len(k1),
        "n_visible0": int(w0.visible.sum()), "n_visible1": int(w1.visible.sum()),
        "n_geometry_evaluable0": int(w0.evaluable.sum()),
        "n_geometry_evaluable1": int(w1.evaluable.sum()),
        "n_putative": 0, "n_evaluable": 0, "n_correct": 0,
    }
    counts["n_covisible"] = min(counts["n_visible0"], counts["n_visible1"])
    # This measures detector repeatability independently of descriptor quality.
    repeated0, _ = mutual_pairs(k0, k1, w0, w1, threshold)
    counts["n_repeatable"] = len(repeated0)
    if not len(d0) or not len(d1):
        return counts
    similarity = d0 @ d1.T
    row, col = similarity.argmax(1), similarity.argmax(0)
    i0 = torch.arange(len(d0), device=d0.device)
    i0 = i0[col[row] == i0]
    i1 = row[i0]
    visible = w0.visible[i0] & w1.visible[i1]
    # A known occlusion/out-of-frame projection disproves a putative match even
    # when reverse geometry is missing. Otherwise require both directions.
    evaluable = visible | w0.outside[i0] | w1.outside[i1]
    error01 = torch.linalg.vector_norm(w0.points[i0] - k1[i1], dim=-1)
    error10 = torch.linalg.vector_norm(w1.points[i1] - k0[i0], dim=-1)
    correct = visible & (error01 <= threshold) & (error10 <= threshold)
    counts.update(n_putative=len(i0), n_evaluable=int(evaluable.sum()),
                  n_correct=int(correct.sum()))
    return counts


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def validate(model, datasets, config, device):
    previous_mode = model.training
    model.eval()
    metrics, selections = {}, []
    selection = config.get("validation_selection_metric", "mean_correct_matches")
    if selection not in ("mean_correct_matches", "mma3", "ms3"):
        model.train(previous_mode)
        raise ValueError(f"Unknown validation_selection_metric: {selection}")
    seed = config.get("validation_seed", config.get("seed", 0))
    try:
        for name, dataset in datasets.items():
            indices = validation_indices(dataset, config["validation_max_pairs"], seed)
            # DataLoader allocates a base seed even with zero workers. Give it
            # its own generator so periodic validation cannot perturb training.
            generator = torch.Generator().manual_seed(seed)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False,
                num_workers=0, collate_fn=collate_pairs, generator=generator)
            totals = {}
            empty_with_gt = False
            for batch in loader:
                sample = to_device(batch[0], device)
                p0 = model(sample["image0"][None], add_random=False, threshold=True)
                p1 = model(sample["image1"][None], add_random=False, threshold=True)

                def pixels(pred, image):
                    h, w = image.shape[-2:]
                    return (pred["keypoints"][0] + 1) / 2 * image.new_tensor([w-1, h-1])

                counts = pair_statistics(
                    pixels(p0, sample["image0"]), pixels(p1, sample["image1"]),
                    p0["descriptors"][0], p1["descriptors"][0],
                    sample["warp01"], sample["warp10"])
                if not counts["n_putative"]:
                    warp = sample["warp01"]
                    empty_with_gt |= warp["mode"] == "homo" or all(
                        bool((torch.isfinite(warp[key]) & (warp[key] > 0)).any())
                        for key in ("depth0", "depth1"))
                for key, value in counts.items():
                    totals[key] = totals.get(key, 0) + value
            n = len(indices)
            if not n:
                raise ValueError(f"Validation dataset {name} has no pairs")
            result = dict(totals, n_pairs=n)
            result.update(
                mma3=_ratio(totals["n_correct"], totals["n_evaluable"]),
                ms3=_ratio(totals["n_correct"], totals["n_covisible"]),
                repeatability3=_ratio(totals["n_repeatable"], totals["n_covisible"]),
                gt_coverage=_ratio(totals["n_evaluable"], totals["n_putative"]),
                keypoint_gt_coverage=_ratio(
                    totals["n_geometry_evaluable0"] + totals["n_geometry_evaluable1"],
                    totals["n_keypoints0"] + totals["n_keypoints1"]),
                mean_matches=totals["n_putative"] / n,
                mean_correct_matches=totals["n_correct"] / n,
                mean_keypoints0=totals["n_keypoints0"] / n,
                mean_keypoints1=totals["n_keypoints1"] / n)
            metrics.update({f"{name}/{key}": value for key, value in result.items()})
            if totals["n_evaluable"] and result[selection] is not None:
                selections.append(result[selection])
            elif not totals["n_putative"] and empty_with_gt:
                # A detector returning no matches must score zero, not disappear
                # from source averaging. MMA itself remains undefined.
                selections.append(0.0)
            else:
                raise ValueError(f"Validation source {name} has no evaluable matches for "
                                 f"{selection}; check ground-truth coverage and holdout geometry")
        if not selections:
            raise ValueError("Validation has no evaluable matches for checkpoint selection; "
                             "check detected keypoints, ground-truth coverage and holdout geometry")
        metrics["selection_metric"] = sum(selections) / len(selections)
        return metrics
    finally:
        model.train(previous_mode)
