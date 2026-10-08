"""Sparse DKD: discrete selection, differentiable local soft argmax.

The public DKD dispersity uses squared, radius-normalized distances; Eq. (8)
prints an unsquared distance. Both are selectable, with the public code default.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.soft_detect import simple_nms


def integer_patches(fmap, points, size):
    """Match custom_ops forward, with autograd accumulating overlapping patches."""
    radius = size // 2
    padded = F.pad(fmap, (radius, radius, radius, radius))
    delta = torch.arange(size, device=fmap.device)
    y, x = torch.meshgrid(delta, delta, indexing="ij")
    points = points.long()
    return padded[:, points[:, 1, None, None] + y,
                  points[:, 0, None, None] + x].permute(1, 0, 2, 3)


def point_nms(points, scores, radius):
    """Score-ordered suppression in pixel coordinates; keep original ordering."""
    if not len(points):
        return torch.empty(0, device=points.device, dtype=torch.long)
    distances = torch.cdist(points.detach(), points.detach())
    blocked = torch.zeros(len(points), device=points.device, dtype=torch.bool)
    keep = []
    for index in scores.detach().argsort(descending=True, stable=True):
        if not blocked[index]:
            keep.append(index)
            blocked |= distances[index] < radius
    return torch.stack(keep).sort().values


class TrainingDetector(nn.Module):
    def __init__(self, radius=2, top_k=400, scores_th=0.2, n_limit=5000,
                 random_k=400, temperature=0.1, peak_mode="upstream"):
        super().__init__()
        self.radius, self.top_k = radius, top_k
        self.scores_th, self.n_limit = scores_th, n_limit
        self.random_k, self.temperature, self.peak_mode = random_k, temperature, peak_mode

    def forward(self, score_map, *, add_random=True, threshold=False):
        _, _, h, w = score_map.shape
        r = self.radius
        if min(h, w) <= 2 * r + 1:
            raise ValueError("Image is smaller than the DKD window")
        result = {k: [] for k in ("keypoints", "scores", "score_dispersity", "num_detected")}
        delta = torch.arange(-r, r + 1, device=score_map.device, dtype=score_map.dtype)
        gy, gx = torch.meshgrid(delta, delta, indexing="ij")
        grid = torch.stack((gx, gy), -1).reshape(-1, 2)
        wh = score_map.new_tensor([w - 1, h - 1])
        for scores in score_map:
            detached = scores.detach()[0]
            nms_scores = simple_nms(scores.detach()[None], r)[0, 0]
            nms_scores[:r] = 0
            nms_scores[-r:] = 0
            nms_scores[:, :r] = 0
            nms_scores[:, -r:] = 0
            # Do not fill the top-k quota with suppressed zero-score pixels.
            indices = (nms_scores > 0).flatten().nonzero().flatten()
            if threshold:
                indices = indices[detached.flatten()[indices] > self.scores_th]
            limit = self.n_limit if threshold else self.top_k
            indices = indices[detached.flatten()[indices].argsort(descending=True, stable=True)[:limit]]
            centers = torch.stack((indices % w, indices // w), -1)
            if len(centers):
                patches = integer_patches(scores, centers, 2*r+1).flatten(1)
                probability = (patches / self.temperature).softmax(-1)
                residual = probability @ grid
                distances = torch.linalg.vector_norm(grid[None] - residual[:, None], dim=-1)
                if self.peak_mode == "upstream":
                    distances = (distances / r).square()
                elif self.peak_mode != "paper":
                    raise ValueError("peak_mode must be upstream or paper")
                dispersity = (probability * distances).sum(-1)
                points = centers.to(scores) + residual
            else:
                points = scores.new_empty((0, 2))
                dispersity = scores.new_empty(0)
            n_detected = len(points)
            if add_random and self.random_k:
                # Random probes train score and descriptor heads, but have no DKD peak loss.
                random = torch.rand(self.random_k, 2, device=scores.device, dtype=scores.dtype)
                random = random * (wh - 2*r) + r
                points = torch.cat((points, random))
            normalized = points / wh * 2 - 1
            sampled = F.grid_sample(scores[None], normalized[None, None], align_corners=True)[0, 0, 0]
            keep = point_nms(points, sampled, r)
            detected_keep = keep[keep < n_detected]
            result["keypoints"].append(normalized[keep])
            result["scores"].append(sampled[keep])
            result["score_dispersity"].append(dispersity[detected_keep])
            result["num_detected"].append(len(detected_keep))
        return result
