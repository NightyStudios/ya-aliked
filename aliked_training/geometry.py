"""Pixel-space warps for homographies and DISK COLMAP cameras.

Depth samples and visibility tests are discrete supervision. Projection remains
differentiable with respect to keypoints, as in ALIKE's warp_se3.
"""
from dataclasses import dataclass
import torch


@dataclass
class Warp:
    points: torch.Tensor
    visible: torch.Tensor
    outside: torch.Tensor

    @property
    def evaluable(self):
        """Known visible or definitely outside; depth holes remain unknown."""
        return self.visible | self.outside

    @property
    def unknown(self):
        return ~self.evaluable


def in_bounds(points, shape):
    h, w = shape
    return (torch.isfinite(points).all(-1) & (points[:, 0] >= 0) &
            (points[:, 0] <= w-1) & (points[:, 1] >= 0) & (points[:, 1] <= h-1))


def depth_at(points, depth):
    """Bilinear depth, only if all four supporting pixels have positive depth."""
    points = points.detach()
    h, w = depth.shape
    finite = torch.nan_to_num(points, nan=-1., posinf=-1., neginf=-1.)
    x, y = finite[:, 0].clamp(0, w-1), finite[:, 1].clamp(0, h-1)
    x0, y0, x1, y1 = x.floor().long(), y.floor().long(), x.ceil().long(), y.ceil().long()
    d00, d01, d10, d11 = depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]
    wx, wy = x-x0, y-y0
    d = d00*(1-wx)*(1-wy) + d01*wx*(1-wy) + d10*(1-wx)*wy + d11*wx*wy
    valid = in_bounds(points, depth.shape) & torch.stack((d00, d01, d10, d11), -1).gt(0).all(-1)
    return d, valid & torch.isfinite(d)


def warp_points(points, params):
    if params["mode"] == "homo":
        homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), -1)
        projected = homogeneous @ params["H"].T
        z = projected[:, 2]
        good_z = z.abs() > 1e-8
        xy = projected[:, :2] / torch.where(good_z, z, torch.ones_like(z))[:, None]
        valid = good_z & in_bounds(xy.detach(), params["shape1"])
        return Warp(xy, valid, ~valid)
    if params["mode"] != "se3":
        raise ValueError("Unknown warp mode")
    d0, valid0 = depth_at(points, params["depth0"])
    homogeneous = torch.cat((points + 0.5, torch.ones_like(points[:, :1])), -1)
    xyz = (homogeneous @ torch.linalg.inv(params["K0"]).T) * d0[:, None]
    xyz1 = xyz @ params["T01"][:3, :3].T + params["T01"][:3, 3]
    projected = xyz1 @ params["K1"].T
    z = xyz1[:, 2]
    safe_z = torch.where(z.abs() > 1e-8, z, torch.ones_like(z))
    xy = projected[:, :2] / safe_z[:, None] - 0.5
    bounds = in_bounds(xy.detach(), params["depth1"].shape) & (z.detach() > 0)
    d1, valid1 = depth_at(xy, params["depth1"])
    consistent = (z.detach() - d1).abs() < params.get("depth_tolerance", 0.05)
    visible = valid0 & bounds & valid1 & consistent
    outside = valid0 & (~bounds | (valid1 & ~consistent))
    # Unknown depths are neither positives nor definite negatives.
    return Warp(xy, visible.detach(), outside.detach())


def mutual_pairs(p0, p1, warp01, warp10, threshold):
    """ALIKE-style mutual nearest neighbours of symmetric reprojection distance."""
    ids0 = warp01.visible.nonzero().flatten()
    ids1 = warp10.visible.nonzero().flatten()
    if not len(ids0) or not len(ids1):
        empty = torch.empty(0, device=p0.device, dtype=torch.long)
        return empty, empty
    distances = (torch.cdist(p0[ids0], warp10.points[ids1]) +
                 torch.cdist(warp01.points[ids0], p1[ids1])) / 2
    row = distances.detach().argmin(1)
    col = distances.detach().argmin(0)
    i = torch.arange(len(ids0), device=p0.device)
    valid = (col[row] == i) & (distances.detach()[i, row] < threshold)
    return ids0[valid], ids1[row[valid]]
