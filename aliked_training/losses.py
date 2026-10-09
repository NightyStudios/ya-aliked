"""ALIKED Sec. V (7)-(14); no dense descriptor probability map or dustbin.

Reliability uses the matching-location scalar clarified by the author in #11.
The reliability target is detached to train the score head, following ALIKE.
"""
import torch
from torch.nn import functional as F
from .geometry import warp_points, mutual_pairs


def sparse_nre(d0, d1, i0, i1, temperature):
    if not len(i0):
        return (d0.sum() + d1.sum()) * 0
    logits = d0 @ d1.T / temperature
    # All target descriptors (also random probes) are in the denominator.
    return (F.cross_entropy(logits[i0], i1) +
            F.cross_entropy(logits.T[i1], i0)) / 2


def reliable_loss(d0, d1, s0, s1, i0, i1, temperature, *, return_targets=False):
    if not len(i0):
        loss = (s0.sum() + s1.sum()) * 0
        return (loss, s0.detach()[:0]) if return_targets else loss
    with torch.no_grad():
        logits = d0.detach() @ d1.detach().T / temperature
        r0 = logits[i0].softmax(-1)[torch.arange(len(i0), device=i0.device), i1]
        r1 = logits.T[i1].softmax(-1)[torch.arange(len(i1), device=i1.device), i0]
    scores0, scores1 = s0[i0], s1[i1]
    a = ((1-r0) * scores0).sum() / scores0.sum().clamp_min(1e-8)
    b = ((1-r1) * scores1).sum() / scores1.sum().clamp_min(1e-8)
    loss = (a + b) / 2
    return (loss, torch.cat((r0, r1))) if return_targets else loss


def pair_loss(pred0, pred1, batch, config, *, diagnostics=False):
    """Return the original loss keys; optionally add detached scalar diagnostics.

    Ratios pool points, counts average per pair/image, and source losses use the
    same weighting as the main losses. Empty populations report zero plus counts.
    No diagnostic consumes randomness or synchronizes individual CUDA scalars.
    """
    zero = (pred0["score_map"].sum() + pred1["score_map"].sum()) * 0
    parts = {key: [] for key in ("rp", "pk", "ds", "re")}
    counts = []
    if diagnostics:
        source_parts, targets = {}, []
        visible, unknown, scores, detected, random, sampling_outside = [], [], [], [], [], []
    for index, sample in enumerate(batch):
        if diagnostics:
            starts = {key: len(values) for key, values in parts.items()}
        h0, w0 = sample["image0"].shape[-2:]
        h1, w1 = sample["image1"].shape[-2:]
        p0 = (pred0["keypoints"][index] + 1) / 2 * zero.new_tensor([w0-1, h0-1])
        p1 = (pred1["keypoints"][index] + 1) / 2 * zero.new_tensor([w1-1, h1-1])
        warp01 = warp_points(p0, sample["warp01"])
        warp10 = warp_points(p1, sample["warp10"])
        i0, i1 = mutual_pairs(p0, p1, warp01, warp10, config["gt_threshold"])
        counts.append(len(i0))
        if len(i0):
            rp = (torch.linalg.vector_norm(p0[i0]-warp10.points[i1], ord=config["reprojection_norm"], dim=-1) +
                  torch.linalg.vector_norm(p1[i1]-warp01.points[i0], ord=config["reprojection_norm"], dim=-1)) / 2
            parts["rp"].append((rp.mean(), len(i0)))
        d0, d1 = pred0["descriptors"][index], pred1["descriptors"][index]
        s0, s1 = pred0["scores"][index], pred1["scores"][index]
        if len(i0):
            parts["ds"].append((sparse_nre(d0, d1, i0, i1, config["t_des"]), len(i0)))
            reliability = reliable_loss(d0, d1, s0, s1, i0, i1, config["t_rel"],
                                        return_targets=diagnostics)
            if diagnostics:
                reliability, target = reliability
                targets.append(target)
            parts["re"].append((reliability, 1))
        for pred in (pred0, pred1):
            dispersity = pred["score_dispersity"][index]
            if len(dispersity):
                parts["pk"].append((dispersity.mean(), len(dispersity)))
        if diagnostics:
            source = str(sample.get("source", "unspecified"))
            buckets = source_parts.setdefault(source, {key: [] for key in parts})
            for key, values in parts.items():
                buckets[key].extend((value.detach(), count) for value, count in values[starts[key]:])
            for pred, points, warp, h, w in ((pred0, p0, warp01, h0, w0),
                                             (pred1, p1, warp10, h1, w1)):
                visible.append(warp.visible.detach())
                unknown.append((~(warp.visible | warp.outside)).detach())
                scores.append(pred["scores"][index].detach())
                n_detected = pred.get("num_detected", [len(p) for p in pred["score_dispersity"]])[index]
                detected.append(n_detected)
                random.append(len(points) - n_detected)
                if "offsets" in pred:
                    positions = points.detach()[:, None] + pred["offsets"][index].detach()
                    inside = torch.isfinite(positions).all(-1) & (positions >= 0).all(-1)
                    inside &= (positions <= positions.new_tensor([w-1, h-1])).all(-1)
                    sampling_outside.append((~inside).flatten())
    losses = {key: sum(value * count for value, count in values) / sum(count for _, count in values)
              if values else zero for key, values in parts.items()}
    losses["total"] = sum(config["weights"][key] * losses[key] for key in parts)
    losses["matches"] = sum(counts)
    if diagnostics:
        base = zero.detach()

        def pooled(values):
            return torch.cat(values).to(base) if values else base.new_empty(0)

        def mean(values):
            return values.mean() if values.numel() else base

        reliability = pooled(targets)
        point_scores = pooled(scores)
        diag = {
            "matches_mean": base.new_tensor(counts).mean() if counts else base,
            "empty_pair_fraction": base.new_tensor(sum(n == 0 for n in counts) / max(len(counts), 1)),
            "visible_ratio": mean(pooled(visible)),
            "unknown_ratio": mean(pooled(unknown)),
            "detected_points_mean": base.new_tensor(detected).mean() if detected else base,
            "random_points_mean": base.new_tensor(random).mean() if random else base,
            "score_mean": mean(point_scores),
            "score_std": point_scores.std(unbiased=False) if point_scores.numel() else base,
            "score_min": point_scores.min() if point_scores.numel() else base,
            "score_max": point_scores.max() if point_scores.numel() else base,
            "reliability_target_mean": mean(reliability),
            "reliability_target_std": reliability.std(unbiased=False) if reliability.numel() else base,
            "reliability_target_count": base.new_tensor(reliability.numel()),
            "sddh_outside_ratio": mean(pooled(sampling_outside)),
            "sddh_samples": base.new_tensor(sum(v.numel() for v in sampling_outside)),
        }
        for source, buckets in source_parts.items():
            for key, values in buckets.items():
                diag[f"source/{source}/{key}"] = (
                    sum(value * count for value, count in values) / sum(count for _, count in values)
                    if values else base)
        losses["diagnostics"] = diag
    return losses
