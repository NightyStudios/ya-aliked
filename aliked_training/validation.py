"""Deterministic holdout MMA@3 on mutual descriptor nearest neighbours."""
import torch
from .data import collate_pairs, to_device
from .geometry import warp_points


@torch.no_grad()
def validate(model, datasets, config, device):
    previous_mode = model.training
    model.eval()
    metrics = {}
    try:
        for name, dataset in datasets.items():
            accuracies, matches = [], []
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                                 num_workers=0, collate_fn=collate_pairs)
            for index, batch in enumerate(loader):
                if index >= config["validation_max_pairs"]:
                    break
                sample = to_device(batch[0], device)
                p0 = model(sample["image0"][None], add_random=False, threshold=True)
                p1 = model(sample["image1"][None], add_random=False, threshold=True)
                d0, d1 = p0["descriptors"][0], p1["descriptors"][0]
                if not len(d0) or not len(d1):
                    accuracies.append(0.0)
                    matches.append(0)
                    continue
                similarity = d0 @ d1.T
                row, col = similarity.argmax(1), similarity.argmax(0)
                i0 = torch.arange(len(d0), device=device)
                valid = col[row] == i0
                i0, i1 = i0[valid], row[valid]
                def pixels(pred, image):
                    h, w = image.shape[-2:]
                    return (pred["keypoints"][0] + 1) / 2 * image.new_tensor([w-1, h-1])
                k0, k1 = pixels(p0, sample["image0"]), pixels(p1, sample["image1"])
                warped = warp_points(k0[i0], sample["warp01"])
                errors = torch.linalg.vector_norm(warped.points - k1[i1], dim=-1)
                correct = warped.visible & (errors <= 3)
                accuracies.append(correct.float().mean().item() if len(correct) else 0.0)
                matches.append(len(correct))
            if not accuracies:
                raise ValueError(f"Validation dataset {name} has no pairs")
            metrics[name + "/mma3"] = sum(accuracies) / len(accuracies)
            metrics[name + "/mean_matches"] = sum(matches) / len(matches)
        # Matching quality on every configured holdout source, equal source weight.
        metrics["selection_metric"] = sum(v for k, v in metrics.items() if k.endswith("/mma3")) / len(datasets)
        return metrics
    finally:
        model.train(previous_mode)
