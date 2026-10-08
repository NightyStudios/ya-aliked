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


def reliable_loss(d0, d1, s0, s1, i0, i1, temperature):
    if not len(i0):
        return (s0.sum() + s1.sum()) * 0
    with torch.no_grad():
        logits = d0.detach() @ d1.detach().T / temperature
        r0 = logits[i0].softmax(-1)[torch.arange(len(i0), device=i0.device), i1]
        r1 = logits.T[i1].softmax(-1)[torch.arange(len(i1), device=i1.device), i0]
    scores0, scores1 = s0[i0], s1[i1]
    a = ((1-r0) * scores0).sum() / scores0.sum().clamp_min(1e-8)
    b = ((1-r1) * scores1).sum() / scores1.sum().clamp_min(1e-8)
    return (a + b) / 2


def pair_loss(pred0, pred1, batch, config):
    zero = (pred0["score_map"].sum() + pred1["score_map"].sum()) * 0
    parts = {key: [] for key in ("rp", "pk", "ds", "re")}
    counts = []
    for index, sample in enumerate(batch):
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
            parts["re"].append((reliable_loss(d0, d1, s0, s1, i0, i1, config["t_rel"]), 1))
        for pred in (pred0, pred1):
            dispersity = pred["score_dispersity"][index]
            if len(dispersity):
                parts["pk"].append((dispersity.mean(), len(dispersity)))
    losses = {key: sum(value * count for value, count in values) / sum(count for _, count in values)
              if values else zero for key, values in parts.items()}
    losses["total"] = sum(config["weights"][key] * losses[key] for key in parts)
    losses["matches"] = sum(counts)
    return losses
