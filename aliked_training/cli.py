"""Explicit inspect/train commands. Inspection never constructs an optimizer."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import load_config
from .data import DISKMegaDepth, HomographicPairs, MixedPairs, collate_pairs, to_device, resolve
from .model import TrainableALIKED
from .losses import pair_loss
from .validation import validate


def datasets_from_config(config):
    if not config["megadepth_root"] or not config["homography_manifest"]:
        raise ValueError("Set megadepth_root and homography_manifest to existing local/server paths")
    perspective = DISKMegaDepth(config["megadepth_root"], size=config["image_size"],
                               pairs_per_scene=config["pairs_per_scene"], seed=config["seed"],
                               excluded_scenes=config["excluded_scenes"], manifest=config["megadepth_manifest"])
    homographic = HomographicPairs(config["homography_manifest"], config["homography_root"],
                                  size=config["image_size"], seed=config["seed"],
                                  tilt=config["homography_tilt"], rotation=config["rotation_degrees"])
    train = MixedPairs(perspective, homographic, config["homography_probability"], config["seed"])
    validation = {}
    if config["validation_megadepth_root"]:
        if Path(config["validation_megadepth_root"]).resolve() == Path(config["megadepth_root"]).resolve():
            raise ValueError("Validation MegaDepth must have a separate root from training data")
        validation["megadepth"] = DISKMegaDepth(config["validation_megadepth_root"],
                                                size=config["image_size"], augment=False,
                                                pairs_per_scene=1, seed=config["seed"])
        overlapping = {name for name, _ in perspective.scenes} & {name for name, _ in validation["megadepth"].scenes}
        if overlapping:
            raise ValueError(f"Training and validation scenes overlap: {sorted(overlapping)}")
    if config["validation_homography_manifest"]:
        if Path(config["validation_homography_manifest"]).resolve() == Path(config["homography_manifest"]).resolve():
            raise ValueError("Homography validation must have a separate manifest")
        validation["homography"] = HomographicPairs(config["validation_homography_manifest"],
                                                    size=config["image_size"], augment=False)
        if any("image1" not in row for row in validation["homography"].records):
            raise ValueError("Validation homographies require real image0/image1 pairs with known H01")
        def image_paths(dataset):
            return {resolve(dataset.root, row[key]).resolve()
                    for row in dataset.records for key in ("image0", "image1") if key in row}
        if image_paths(homographic) & image_paths(validation["homography"]):
            raise ValueError("Training and validation homography images overlap")
    if not validation:
        raise ValueError("Configure a held-out validation source for selecting the best model")
    return train, validation


def inspect(config, check_data=False):
    report = {"config": config,
              "effective_batch_size": config["batch_size"] * config["accumulate_batches"],
              "optimizer_steps": config["max_steps"],
              "microbatches": config["max_steps"] * config["accumulate_batches"],
              "training_started": False,
              "assumptions": ["100K means optimizer updates", "lr/warmup inherited from ALIKE",
                              "mixed source ratio is configurable, default 1:1",
                              "public-code DKD peak convention by default"]}
    if check_data:
        train, validation = datasets_from_config(config)
        # Decode actual images/depth/calibration; do not instantiate or run a network.
        for name, dataset in {"train/megadepth": train.perspective,
                              "train/homography": train.homographic, **validation}.items():
            sample = dataset[0]
            report[name] = {"pairs": len(dataset), "image_shape": list(sample["image0"].shape),
                            "geometry": sample["warp01"]["mode"]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def checkpoint_state(model, optimizer, scheduler, config, step, epoch, batch_index, best):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "config": config, "step": step,
            "epoch": epoch, "batch_index": batch_index, "best_metric": best,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, resume=None):
    """Full training entry point, supplied as code; not run during reconstruction."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    train_data, validation = datasets_from_config(config)
    batches_per_epoch = len(train_data) // config["batch_size"]
    if not batches_per_epoch:
        raise ValueError("Dataset is smaller than one training batch")
    model = TrainableALIKED(config, config["initial_weights"]).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["adam_betas"]))
    warmup = config["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s+1)/max(warmup, 1), 1.0))
    step, epoch, batch_index, best = 0, 0, 0, -math.inf
    if resume:
        # Full checkpoints include Python/NumPy RNG state. Load only a trusted local checkpoint.
        state = torch.load(resume, map_location=device, weights_only=False)
        changes = [key for key in config if state["config"][key] != config[key]
                   and key not in ("output_dir", "device", "max_steps", "workers")]
        if changes:
            raise ValueError(f"Resume configuration differs: {changes}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, batch_index = state["step"], state["epoch"], state["batch_index"]
        best = state["best_metric"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    optimizer.zero_grad(set_to_none=True)
    microbatches = 0
    totals = {key: 0.0 for key in ("rp", "pk", "ds", "re", "total", "matches")}
    while step < config["max_steps"]:
        train_data.set_epoch(epoch)
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        indices = torch.randperm(len(train_data), generator=generator).tolist()
        indices = indices[batch_index*config["batch_size"]:batches_per_epoch*config["batch_size"]]
        loader = DataLoader(train_data, batch_size=config["batch_size"], sampler=indices,
                            num_workers=config["workers"], drop_last=True, collate_fn=collate_pairs,
                            worker_init_fn=seed_worker, generator=generator, pin_memory=device.type == "cuda")
        for batch in loader:
            batch = to_device(batch, device)
            images0 = torch.stack([sample["image0"] for sample in batch])
            images1 = torch.stack([sample["image1"] for sample in batch])
            pred0, pred1 = model(images0), model(images1)
            losses = pair_loss(pred0, pred1, batch, config)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite loss at update {step}, batch {batch_index}")
            (losses["total"] / config["accumulate_batches"]).backward()
            for key in totals:
                value = losses[key]
                totals[key] += float(value.detach()) if isinstance(value, torch.Tensor) else value
            microbatches += 1
            batch_index += 1
            if microbatches < config["accumulate_batches"]:
                continue
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite gradients at update {step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "lr": optimizer.param_groups[0]["lr"],
                      **{key: value/microbatches for key, value in totals.items()}}
            if step % config["validation_interval"] == 0 or step == config["max_steps"]:
                # Holdout does not consume the RNG used for future random training probes.
                metrics = validate(model, validation, config, device)
                record.update(metrics)
                improved = metrics["selection_metric"] > best
                best = max(best, metrics["selection_metric"])
                state = checkpoint_state(model, optimizer, scheduler, config, step, epoch, batch_index, best)
                save_checkpoint(output / "last.pt", state)
                if improved:
                    save_checkpoint(output / "best.pt", state)
                    # Same state-dict format as upstream inference models.
                    save_checkpoint(output / "best-model.pth", model.state_dict())
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            microbatches = 0
            totals = dict.fromkeys(totals, 0.0)
            if step >= config["max_steps"]:
                break
        if batch_index >= batches_per_epoch:
            epoch += 1
            batch_index = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "inspect":
        if args.resume:
            parser.error("--resume is only for train")
        inspect(config, args.check_data)
    else:
        if args.check_data:
            parser.error("Use inspect --check-data to validate files")
        train(config, args.resume)


if __name__ == "__main__":
    main()
